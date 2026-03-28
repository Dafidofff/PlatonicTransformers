# ImageNet Experiments Tracker

## Overview
ImageNet-1K classification experiments using the Platonic Transformer architecture.
Comparing baseline (trivial group) against equivariant models (flop_2d_2, cyclic_4)
with DeiT-III aligned training recipe.

**Reference papers:**
- DeiT III: Revenge of the ViT (Touvron et al., 2022) — [arxiv:2204.07118](https://arxiv.org/abs/2204.07118)
- Flopping for FLOPs (Bökman et al., 2025) — [arxiv:2502.05169](https://arxiv.org/abs/2502.05169)
- An Image is Worth 16x16 Words (Dosovitskiy et al., 2021) — [arxiv:2010.11929](https://arxiv.org/abs/2010.11929)

PDFs stored in `workspaces/platonic-imagenet/`.

## Environment
- **Cluster:** Snellius (SURF), H100 GPUs (95GB)
- **Python:** 3.12.4 via uv venv
- **Key packages:** PyTorch 2.4.0+cu124, DALI 2.0, quack-kernels 0.3.3
- **CUDA:** 12.8 (module load)
- **W&B project:** Platonic-ImageNet

## Current Runs — v3 (2026-03-22)

DeiT-III aligned hyperparameters, 4×H100 with torch.compile.

| Job ID | Solid | Params | hidden_dim | heads | layers | epochs | eff. bs | GPUs | Status |
|--------|-------|--------|-----------|-------|--------|--------|---------|------|--------|
| 21046799 | trivial_2 | 79.4M | 768 | 12 | 12 | 400 | 2048 | 4×H100 | Running |
| 21046800 | flop_2d_2 | 40.4M | 768 | 12 | 12 | 400 | 2048 | 4×H100 | Running |
| 21046801 | cyclic_4 | ~40M | 768 | 12 | 12 | 400 | 2048 | 4×H100 | Running |

### v3 Training Recipe (matches DeiT-III Table 1 "Ours ImNet-1k")

| Parameter | Value |
|-----------|-------|
| Optimizer | LAMB (timm) |
| Learning rate | 3e-3 |
| LR schedule | Cosine |
| Weight decay | 0.02 |
| Warmup epochs | 5 |
| Epochs | 400 |
| Batch size (per GPU) | 512 |
| Effective batch size | 2048 (4 GPUs × 512) |
| Loss | BCE |
| Label smoothing | 0 |
| Stochastic depth | 0.1 |
| LayerScale init | 1e-4 |
| Mixup alpha | 0.8 |
| CutMix alpha | 1.0 |
| Data augmentation | 3-Augment (grayscale, solarize, blur) + ColorJitter 0.3 |
| RandAugment | Disabled |
| Random erasing | 0.25 |
| Eval crop ratio | 1.0 |
| Resolution | 224×224, patch 16 |
| Precision | bf16-mixed |
| Positional encoding | RoPE σ=16.0, APE σ=16.0, spiral freq init, learned |

### Hardware Optimizations
- `torch.backends.cudnn.benchmark = True`
- `torch.backends.cuda.enable_flash_sdp(True)`
- `torch.compile` on model.net
- `torch.set_float32_matmul_precision('medium')`

### Known Differences from DeiT-III
- **Architecture:** Platonic Transformer (point-cloud patch embedding + group machinery), not standard ViT
- **Repeated Augmentation:** Not implemented (DeiT-III uses it, ~0.5-1pp impact)
- **Optimizer:** timm Lamb, not Apex FusedLAMB (Apex not installed)

## Speed Benchmarks (2026-03-22)

3-epoch benchmarks on trivial_2 (79.4M params), all with effective bs=2048.

| Config | GPUs | Epoch 1 | Epoch 2 | Epoch 3 | Notes |
|--------|------|---------|---------|---------|-------|
| 4GPU, hwopt + compile | 4×H100 | 12.3 min | 9.9 min | **9.5 min** | Winner. Epoch 1 slow due to compile |
| 4GPU, hwopt | 4×H100 | 11.3 min | 11.1 min | 10.8 min | Consistent, no compile overhead |
| 1GPU, hwopt + compile | 1×H100 | 26.6 min | — | — | Partial (cancelled) |
| 1GPU, hwopt | 1×H100 | 32.9 min | — | — | Partial (cancelled) |
| 1GPU, no opts | 1×H100 | 32.4 min | — | — | Partial (cancelled) |

**Conclusion:** 4GPU + compile is ~3.3x faster than 1GPU baseline.
400 epochs ETA: ~63 hours (~2.6 days), fits within 5-day SLURM limit.

## DDP Fix (2026-03-22)

4-GPU DDP was broken: DALI pipelines were created before DDP initialized, so all ranks
got `num_shards=1, shard_id=0` (all reading full dataset on GPU 0 → NCCL deadlock).

**Fix:** Introduced `ImageNetDataModule` (LightningDataModule) that defers DALI pipeline
creation to `setup()`, which runs after DDP spawns workers. Also:
- Changed DALI train `LastBatchPolicy.DROP` → `FILL` (equal batch count across shards)
- Added `sync_dist=True` to all validation/test `self.log()` calls
- Added `use_distributed_sampler=False` (DALI handles sharding)
- Disabled quack cross_entropy in multi-GPU mode (stride incompatibility with DDP)

## v2 Runs (2026-03-20, completing)

Positional encoding tuned (rope_σ=16, ape_σ=16, spiral freq init), but suboptimal
training recipe (AdamW, lr=8e-4, bs=256, 100 epochs, 1 GPU).

| Job ID | W&B Name | Solid | Epoch | Val Top-1 | Val Top-5 |
|--------|----------|-------|-------|-----------|-----------|
| 20997867 | treasured-surf-12 | trivial_2 | 81/100 | 73.4% | 91.4% |
| 20997868 | avid-breeze-12 | flop_2d_2 | 79/100 | 70.5% | 89.6% |

## v1 Runs (2026-03-19, completed)

First full 100-epoch runs. rope_σ=1.0, ape_σ=10.0, random freq init.

| W&B Name | Solid | Val Top-1 | Val Top-5 |
|----------|-------|-----------|-----------|
| volcanic-leaf-10 | trivial_2 | 75.9% | 92.8% |
| misty-thunder-10 | cyclic_4 | 66.3% | 87.5% |

## Exploratory Runs (2026-03-18)

- Tested 4-GPU DDP vs 1-GPU: 1-GPU was faster (before DDP fix)
- Tested batch_size 256 vs 1024: similar wall-clock, 256 had better convergence
- Tested torch.compile: caused NCCL issues in multi-GPU (before DDP fix)
- Early runs used hidden_dim=384 (DeiT-Small), switched to 768 (DeiT-Base)

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/run_imagenet_trivial_v3.sh` | **Current:** 4-GPU trivial_2, DeiT-III recipe + compile |
| `scripts/run_imagenet_flop2d2_v3.sh` | **Current:** 4-GPU flop_2d_2, DeiT-III recipe + compile |
| `scripts/run_imagenet_cyclic4_v3.sh` | **Current:** 4-GPU cyclic_4, DeiT-III recipe + compile |
| `scripts/bench_*.sh` | Speed benchmark scripts (3 epochs each) |
| `scripts/test_ddp_4gpu.sh` | DDP verification script (2 epochs, limited batches) |
| `scripts/run_imagenet_trivial_v2.sh` | v2: 1-GPU, tuned PE, old recipe |
| `scripts/run_imagenet_flop2d2.sh` | v2: 1-GPU flop_2d_2, tuned PE, old recipe |

## Setup Notes
- DALI requires `DALI_NO_MMAP=1` on shared filesystems
- Lightning DDP: use `ntasks=1` + `unset SLURM_NTASKS/SLURM_NTASKS_PER_NODE`
- Data at `/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder`
- Checkpoints saved to `mains/logs/Platonic-ImageNet/<wandb_run_id>/checkpoints/`
- Resume: `RESUME_CKPT=/path/to/last.ckpt sbatch scripts/run_imagenet_*.sh`

---

## v4 Runs (2026-03-28) — DeiT-III Fix

**Branch:** `feature/deit3-v4-fixes`

### Changes from v3
1. **random_erasing_prob=0.0** — was leaking 0.25 from YAML default, now explicitly overridden in CLI
2. **gradient_as_bucket_view=True** in DDPStrategy — reduces memory copies during gradient all-reduce
3. No other recipe changes. Same 3-Augment, BCE, LAMB lr=3e-3, etc.

### Known remaining gaps vs DeiT-III (83.1%)
- Repeated Augmentation: not implemented (~unknown impact, no ablation in DeiT-III paper)
- Mean pooling vs CLS token: architectural difference (~0.5-1pp)
- timm Lamb vs Apex FusedLAMB: minor (~0.1-0.3pp)

### v4 trivial_2 run
- **W&B project:** Platonic-ImageNet-v4
- **W&B run:** earthy-galaxy-3 (kmkaxifd), resumed as job 21277828
- **Config:** 4×H100, bs=512 (eff 2048), 400 epochs, compile=true
- **Status:** Running, resumed from epoch 12
- **Early results:** +0.81pp ahead of v3 at epoch 12

### How to launch flop_2d_2 v4 (on any server with DALI + ImageNet)

```bash
# Ensure you are on branch feature/deit3-v4-fixes
git checkout feature/deit3-v4-fixes

# The key changes from v3 are:
#   --augmentation.random_erasing_prob=0.0
#   gradient_as_bucket_view=True in mains/main_imagenet.py DDPStrategy

# Launch flop_2d_2 (adjust data_dir and gpus for your server):
python mains/main_imagenet.py \
    --config configs/imagenet_dali.yaml \
    --dataset.data_dir=/path/to/imagenet/torchvision_ImageFolder \
    --dataset.eval_crop_ratio=1.0 \
    --augmentation.rand_augment="" \
    --augmentation.random_erasing_prob=0.0 \
    --mixup.label_smoothing=0.0 \
    --training.epochs=400 \
    --training.batch_size=512 \
    --model.solid_name=flop_2d_2 \
    --model.hidden_dim=768 --model.num_heads=12 --model.num_layers=12 \
    --model.freq_init=spiral --model.learned_freqs=true \
    --model.drop_path_rate=0.1 --model.layer_scale_init_value=1e-4 \
    --model.attention=true --model.dense_mode=true \
    --model.ffn_readout=false --model.use_key=false \
    --optimizer.name=lamb --optimizer.lr=3e-3 --optimizer.weight_decay=0.02 \
    --scheduler.warmup_epochs=5 \
    --system.gpus=4 \
    --system.cudnn_benchmark=true --system.flash_sdp=true --system.compile=true \
    --logging.enabled=true --logging.project_name=Platonic-ImageNet-v4
```

### Also consider: flop_2d_2_wide (1088, ~79.5M params)
Same command but with:
```
    --model.solid_name=flop_2d_2 \
    --model.hidden_dim=1088 --model.num_heads=16 \
```
