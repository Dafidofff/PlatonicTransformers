#!/bin/bash
#SBATCH --job-name=ddp-test-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail

# ─── Environment ─────────────────────────────────────────────────────────────
source ~/imagenet/PlatonicTransformers/.venv/bin/activate

module load 2025
module load CUDA/12.8.0

CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

export DALI_NO_MMAP=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Prevent Lightning from auto-detecting SLURM task layout
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

# NCCL and DDP debug for troubleshooting
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_TIMEOUT=300

# ─── Verify ──────────────────────────────────────────────────────────────────
echo "=== DDP 4-GPU Test ==="
echo "Node:  $(hostname)"
echo "GPUs:  $(nvidia-smi -L)"
echo "Job:   $SLURM_JOB_ID"
echo "Date:  $(date)"

# ─── Run ─────────────────────────────────────────────────────────────────────
cd ~/imagenet/PlatonicTransformers
mkdir -p logs

# Minimal 3-epoch run to verify DDP works end-to-end
# Uses small batch size and no grad accumulation for speed
python mains/main_imagenet.py \
    --config configs/imagenet_dali.yaml \
    --dataset.data_dir=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder \
    --dataset.eval_crop_ratio=1.0 \
    --augmentation.rand_augment="" \
    --augmentation.use_three_augment=false \
    --augmentation.color_jitter=0.0 \
    --augmentation.random_erasing_prob=0.0 \
    --mixup.mixup_alpha=0.0 \
    --mixup.cutmix_alpha=0.0 \
    --mixup.label_smoothing=0.0 \
    --training.epochs=2 \
    --training.batch_size=256 \
    --training.accumulate_grad_batches=1 \
    --training.loss_fn=ce \
    --training.limit_train_batches=50 \
    --training.limit_val_batches=20 \
    --model.solid_name=trivial_2 \
    --model.hidden_dim=384 \
    --model.num_heads=8 \
    --model.num_layers=6 \
    --model.freq_init=spiral \
    --model.learned_freqs=true \
    --model.drop_path_rate=0.0 \
    --model.layer_scale_init_value=null \
    --model.attention=true \
    --model.dense_mode=true \
    --model.ffn_readout=false \
    --model.use_key=false \
    --optimizer.name=adamw \
    --optimizer.lr=1e-3 \
    --optimizer.weight_decay=0.05 \
    --scheduler.warmup_epochs=1 \
    --system.gpus=4 \
    --logging.enabled=false \
    2>&1

echo ""
echo "=== DDP Test Complete (exit code: $?) ==="
echo "Date: $(date)"
