#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --partition=gpu_h100
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --signal=B:SIGUSR1@120

set -eo pipefail

# ─── Environment ─────────────────────────────────────────────────────────────
source ~/imagenet/PlatonicTransformers/.venv/bin/activate
module load 2025 && module load CUDA/12.8.0
CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_HOME PATH="${CUDA_HOME}/bin:${PATH}" LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export DALI_NO_MMAP=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset SLURM_NTASKS SLURM_NTASKS_PER_NODE

echo "Node: $(hostname) | GPUs: $(nvidia-smi -L | wc -l) | Job: $SLURM_JOB_ID"

# ─── Resume ──────────────────────────────────────────────────────────────────
cd ~/imagenet/PlatonicTransformers && mkdir -p logs
RESUME_ARG=""
if [ -n "$RESUME_CKPT" ]; then
    echo "Resuming from: $RESUME_CKPT"
    RESUME_ARG="--testing.resume_ckpt=$RESUME_CKPT"
fi

# ─── Run ─────────────────────────────────────────────────────────────────────
# v3 final: DeiT-III recipe, 4×H100, torch.compile
python mains/main_imagenet.py \
    --config configs/imagenet_dali.yaml \
    --dataset.data_dir=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder \
    --dataset.eval_crop_ratio=1.0 \
    --augmentation.rand_augment="" \
    --mixup.label_smoothing=0.0 \
    --training.epochs=400 \
    --training.batch_size=512 \
    --model.solid_name=cyclic_4 \
    --model.hidden_dim=768 --model.num_heads=12 --model.num_layers=12 \
    --model.freq_init=spiral --model.learned_freqs=true \
    --model.drop_path_rate=0.1 --model.layer_scale_init_value=1e-4 \
    --model.attention=true --model.dense_mode=true \
    --model.ffn_readout=false --model.use_key=false \
    --optimizer.name=lamb --optimizer.lr=3e-3 --optimizer.weight_decay=0.02 \
    --scheduler.warmup_epochs=5 \
    --system.gpus=4 \
    --system.cudnn_benchmark=true --system.flash_sdp=true --system.compile=true \
    --logging.enabled=true \
    $RESUME_ARG 2>&1
