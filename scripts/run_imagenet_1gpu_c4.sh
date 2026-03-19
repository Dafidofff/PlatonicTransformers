#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --time=2-00:00:00
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

# ─── Verify ──────────────────────────────────────────────────────────────────
echo "Node:  $(hostname)"
echo "GPUs:  $(nvidia-smi -L)"

# ─── Run ─────────────────────────────────────────────────────────────────────
cd ~/imagenet/PlatonicTransformers
mkdir -p logs

python mains/main_imagenet.py \
    --config configs/imagenet_dali.yaml \
    --dataset.data_dir=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder \
    --training.epochs=100 \
    --training.batch_size=256 \
    --model.solid_name=cyclic_4 --model.num_heads=12 \
    --system.gpus=1 \
    --logging.enabled=true 2>&1
