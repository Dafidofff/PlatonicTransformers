import os
import sys
from typing import Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import datetime

import ml_collections
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.strategies import DDPStrategy

# Import model and utilities
from platonic_transformers.utils.config_loader import (
    get_arg_parser,
    load_with_defaults,
    print_config,
)
from platonic_transformers.models.platoformer.platoformer import PlatonicTransformer
from platonic_transformers.models.platoformer.groups import PLATONIC_GROUPS
from platonic_transformers.utils.utils import CosineWarmupScheduler, RandomSOd
from platonic_transformers.utils.callbacks import TimerCallback

# Use quack's fused cross-entropy kernel when available (H100+)
try:
    from quack import cross_entropy as _quack_cross_entropy
except ImportError:
    _quack_cross_entropy = None

# Performance optimization (defaults, can be overridden via config.system)
torch.set_float32_matmul_precision('medium')


class ImageNetDataModule(pl.LightningDataModule):
    """LightningDataModule that creates DALI pipelines AFTER DDP is initialised.

    This is critical for multi-GPU: DALI needs the correct LOCAL_RANK and
    WORLD_SIZE to shard data and place pipelines on the right GPU.  Those
    environment variables are only set after Lightning spawns the DDP workers,
    so the pipelines must be built inside ``setup()`` rather than in ``__init__``.
    """

    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()
        self.config = config
        self._train_loader = None
        self._val_loader = None

    def setup(self, stage: str = None) -> None:
        if self._train_loader is not None:
            return  # already built

        from platonic_transformers.datasets.imagenet_dali import load_data

        self._train_loader, self._val_loader, _ = load_data(self.config)

        # Debug: log iterator lengths per rank
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(
            f"[RANK {local_rank}/{world_size}] train_loader len={len(self._train_loader)}, "
            f"val_loader len={len(self._val_loader)}",
            flush=True,
        )

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._val_loader


class ImageNetModel(pl.LightningModule):
    """LightningModule implementing ImageNet classification with PlatonicTransformer."""

    def __init__(self, config: ml_collections.ConfigDict) -> None:
        super().__init__()
        self.save_hyperparameters({'config': config.to_dict()})
        self.config = config

        # Setup 2D rotation augmentation for point cloud
        self.rotation_generator = RandomSOd(2)

        # ImageNet point cloud: 3 scalar features (RGB) per patch pixel + 2D positions
        patch_size = config.dataset.patch_size
        image_size = getattr(config.dataset, 'final_image_size', config.dataset.image_size)
        in_channels_scalar = patch_size * patch_size * 3
        in_channels_vector = 2

        # Number of patches in the point cloud
        self.avg_num_nodes = (image_size // patch_size) ** 2

        # Validate solid name
        solid_name = config.model.solid_name
        if solid_name not in PLATONIC_GROUPS:
            raise ValueError(
                f"Invalid solid_name '{solid_name}'. "
                f"Must be one of: {list(PLATONIC_GROUPS.keys())}"
            )

        # Initialize Platonic Transformer
        self.net = PlatonicTransformer(
            input_dim=in_channels_scalar,
            input_dim_vec=in_channels_vector,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.dataset.num_classes,
            output_dim_vec=0,
            nhead=config.model.num_heads,
            num_layers=config.model.num_layers,
            solid_name=solid_name,
            spatial_dim=config.model.spatial_dim,
            dense_mode=config.model.dense_mode,
            scalar_task_level=config.model.scalar_task_level,
            vector_task_level=config.model.vector_task_level,
            ffn_readout=config.model.ffn_readout,
            mean_aggregation=config.model.mean_aggregation,
            dropout=config.model.dropout,
            drop_path_rate=config.model.drop_path_rate,
            layer_scale_init_value=config.model.layer_scale_init_value,
            attention=config.model.attention,
            ffn_dim_factor=config.model.ffn_dim_factor,
            norm_type=getattr(config.model, 'norm_type', 'layernorm'),
            rope_sigma=config.model.rope_sigma,
            ape_sigma=config.model.ape_sigma,
            learned_freqs=config.model.learned_freqs,
            freq_init=config.model.freq_init,
            use_key=config.model.use_key,
        )

        # Setup metrics
        num_classes = config.dataset.num_classes
        self.train_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.train_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.valid_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.valid_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)
        self.test_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=1)
        self.test_metric_top5 = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=5)

    def forward(self, data) -> torch.Tensor:
        """Forward pass with optional rotation augmentation."""
        if self.training and self.config.training.train_augm:
            rot = self.rotation_generator().type_as(data.pos)
            data.pos = torch.einsum('ij,bj->bi', rot, data.pos)
        else:
            rot = torch.eye(2, device=data.pos.device)

        vec = rot.transpose(-2, -1).unsqueeze(0).expand(data.pos.shape[0], -1, -1)

        pred, _ = self.net(
            data.x,
            data.pos,
            data.batch,
            vec=vec,
            avg_num_nodes=self.avg_num_nodes,
        )
        return pred

    def _calculate_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculate loss, handling both soft labels (Mixup) and hard labels."""
        if y.ndim == 2:
            # Soft targets from Mixup/CutMix
            return F.binary_cross_entropy_with_logits(pred, y)
        elif self.config.training.loss_fn == "bce":
            y_one_hot = F.one_hot(y, num_classes=self.config.dataset.num_classes).to(dtype=pred.dtype, device=pred.device)
            return F.binary_cross_entropy_with_logits(pred, y_one_hot)
        elif (
            _quack_cross_entropy is not None
            and pred.is_cuda
            and self.config.system.gpus <= 1
        ):
            # quack cross_entropy backward has stride issues with DDP
            return _quack_cross_entropy(pred, y, reduction='mean')
        else:
            return F.cross_entropy(pred, y)

    def training_step(self, data, batch_idx: int) -> torch.Tensor:
        pred = self(data)
        loss = self._calculate_loss(pred, data.y)
        self.log("train_loss", loss, prog_bar=True, batch_size=self.config.training.batch_size)
        # Accuracy only meaningful with hard labels (not Mixup soft targets)
        if data.y.ndim == 1:
            self.train_metric(pred, data.y)
            self.train_metric_top5(pred, data.y)
        return loss

    def validation_step(self, data, batch_idx: int) -> None:
        pred = self(data)
        loss = self._calculate_loss(pred, data.y)
        self.valid_metric(pred, data.y)
        self.valid_metric_top5(pred, data.y)
        self.log("valid_loss", loss, batch_size=self.config.training.batch_size, sync_dist=True)

    def test_step(self, data, batch_idx: int) -> None:
        pred = self(data)
        self.test_metric(pred, data.y)
        self.test_metric_top5(pred, data.y)

    def on_train_epoch_end(self) -> None:
        self.log("train_acc_top1", self.train_metric, prog_bar=True, sync_dist=True)
        self.log("train_acc_top5", self.train_metric_top5, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log("valid_acc_top1", self.valid_metric, prog_bar=True, sync_dist=True)
        self.log("valid_acc_top5", self.valid_metric_top5, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        self.log("test_acc_top1", self.test_metric, prog_bar=True, sync_dist=True)
        self.log("test_acc_top5", self.test_metric_top5, sync_dist=True)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move any straggling CPU tensors (e.g. DALI labels) to the target device."""
        return batch.to(device)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler.

        Splits parameters into decay / no-decay groups following DeiT-III:
        weight decay is NOT applied to biases, LayerScale gammas, learned
        positional-encoding frequencies, or normalization layer weights.
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, module in self.named_modules():
            for pn, _ in module.named_parameters(recurse=False):
                full_name = f"{mn}.{pn}" if mn else pn
                if pn == 'freqs':
                    no_decay.add(full_name)
                elif pn.endswith('bias') or ('layer_scale' in pn) or ('gamma' in pn):
                    no_decay.add(full_name)
                elif pn.endswith('weight') and isinstance(module, whitelist_weight_modules):
                    decay.add(full_name)
                elif pn.endswith('kernel'):
                    decay.add(full_name)
                elif pn.endswith('weight') and isinstance(module, blacklist_weight_modules):
                    no_decay.add(full_name)

        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        missing = param_dict.keys() - (decay | no_decay)
        if missing:
            print(f"Params not explicitly assigned (adding to no_decay): {missing}")
            no_decay.update(missing)

        assert len(decay & no_decay) == 0, f"Overlap: {decay & no_decay}"

        optim_groups = [
            g for g in [
                {"params": [param_dict[n] for n in sorted(decay) if n in param_dict],
                 "weight_decay": self.config.optimizer.weight_decay},
                {"params": [param_dict[n] for n in sorted(no_decay) if n in param_dict],
                 "weight_decay": 0.0},
            ] if g["params"]
        ]
        print(f"Optimizer: {sum(p.numel() for g in optim_groups for p in g['params'])/1e6:.1f}M params "
              f"({sum(p.numel() for p in optim_groups[0]['params'])/1e6:.1f}M decay, "
              f"{sum(p.numel() for p in optim_groups[1]['params'])/1e6:.1f}M no_decay)")

        optimizer_name = self.config.optimizer.name.lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                optim_groups,
                lr=self.config.optimizer.lr,
            )
        elif optimizer_name == "lamb":
            from timm.optim import Lamb
            optimizer = Lamb(
                optim_groups,
                lr=self.config.optimizer.lr,
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {optimizer_name}. Supported: 'adamw', 'lamb'"
            )

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup=self.config.scheduler.warmup_epochs,
            max_iters=self.trainer.max_epochs,
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def main(config: ml_collections.ConfigDict) -> None:
    """Train and evaluate the Platonic Transformer on ImageNet."""

    print_config(config, "ImageNet Training Configuration")

    # Hardware optimizations (configurable via config.system)
    if getattr(config.system, 'cudnn_benchmark', True):
        torch.backends.cudnn.benchmark = True
    if getattr(config.system, 'flash_sdp', True):
        torch.backends.cuda.enable_flash_sdp(True)

    pl.seed_everything(config.seed)

    # Data module defers DALI pipeline creation until after DDP init
    data_module = ImageNetDataModule(config)

    # Configure accelerator
    if config.system.gpus > 0:
        accelerator, devices = "gpu", config.system.gpus
    else:
        raise ValueError("ImageNet training requires GPU (DALI is GPU-only)")

    if config.logging.enabled:
        save_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "logs",
        )
        logger = pl.loggers.WandbLogger(
            project=config.logging.project_name,
            config=config.to_dict(),
            save_dir=save_dir,
        )
    else:
        logger = None

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_last=config.checkpoint.save_last,
        ),
        TimerCallback(),
    ]
    if config.logging.enabled:
        callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))

    # Gradient accumulation (effective_bs = batch_size * gpus * accumulate_grad_batches)
    accumulate_grad_batches = getattr(config.training, 'accumulate_grad_batches', 1)

    # Optional batch limits for fast debugging
    limit_train_batches = getattr(config.training, 'limit_train_batches', 1.0)
    limit_val_batches = getattr(config.training, 'limit_val_batches', 1.0)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=config.system.enable_progress_bar,
        precision=getattr(config.system, 'precision', 'bf16-mixed'),
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            timeout=datetime.timedelta(minutes=30),
        ) if config.system.gpus > 1 else 'auto',
        use_distributed_sampler=False,  # DALI handles sharding internally
        num_sanity_val_steps=0,  # DALI iterators don't support mid-epoch resets
    )

    test_ckpt = config.testing.test_ckpt
    resume_ckpt = getattr(config.testing, 'resume_ckpt', None)
    if test_ckpt is None:
        model = ImageNetModel(config)
        if getattr(config.system, 'compile', False):
            print("Compiling model with torch.compile...")
            model.net = torch.compile(model.net)
        trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)
        trainer.test(model, datamodule=data_module, ckpt_path='best')
    else:
        model = ImageNetModel.load_from_checkpoint(test_ckpt)
        trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    parser = get_arg_parser(default_config_path="configs/imagenet_dali.yaml")
    args, unknown_args = parser.parse_known_args()

    config = load_with_defaults(
        dataset_config=args.config,
        cli_args=unknown_args,
    )

    main(config)
