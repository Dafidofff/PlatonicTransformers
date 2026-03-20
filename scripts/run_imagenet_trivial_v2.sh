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
# v2: Use YAML defaults for rope_sigma (16.0), ape_sigma (16.0), spiral freq init
# Changed from v1: rope_sigma 1.0->16.0, ape_sigma 10.0->16.0, freq_init random->spiral
cd ~/imagenet/PlatonicTransformers
mkdir -p logs

python mains/main_imagenet.py \
    --config configs/imagenet_dali.yaml \
    --dataset.data_dir=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder \
    --training.epochs=100 \
    --training.batch_size=256 \
    --model.solid_name=trivial_2 \
    --model.hidden_dim=768 \
    --model.num_heads=12 \
    --model.num_layers=12 \
    --model.freq_init=spiral \
    --model.learned_freqs=true \
    --model.drop_path_rate=0.1 \
    --model.attention=true \
    --model.dense_mode=true \
    --model.ffn_readout=false \
    --model.use_key=false \
    --optimizer.lr=8e-4 \
    --system.gpus=1 \
    --logging.enabled=true 2>&1
