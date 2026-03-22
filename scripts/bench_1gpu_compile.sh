#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_h100
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail
source ~/imagenet/PlatonicTransformers/.venv/bin/activate
module load 2025 && module load CUDA/12.8.0
CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_HOME PATH="${CUDA_HOME}/bin:${PATH}" LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export DALI_NO_MMAP=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ~/imagenet/PlatonicTransformers && mkdir -p logs
echo "=== BENCH: 1GPU, cudnn.benchmark + flash_sdp + torch.compile ==="
echo "Node: $(hostname), GPUs: $(nvidia-smi -L | head -1)"

python mains/main_imagenet.py \
    --config configs/imagenet_dali.yaml \
    --dataset.data_dir=/scratch-nvme/ml-datasets/imagenet/torchvision_ImageFolder \
    --dataset.eval_crop_ratio=1.0 \
    --augmentation.rand_augment="" \
    --mixup.label_smoothing=0.0 \
    --training.epochs=3 \
    --training.batch_size=256 \
    --training.accumulate_grad_batches=8 \
    --model.solid_name=trivial_2 \
    --model.hidden_dim=768 --model.num_heads=12 --model.num_layers=12 \
    --model.freq_init=spiral --model.learned_freqs=true \
    --model.drop_path_rate=0.1 --model.layer_scale_init_value=1e-4 \
    --model.attention=true --model.dense_mode=true \
    --model.ffn_readout=false --model.use_key=false \
    --optimizer.name=lamb --optimizer.lr=3e-3 --optimizer.weight_decay=0.02 \
    --scheduler.warmup_epochs=5 \
    --system.gpus=1 \
    --system.cudnn_benchmark=true \
    --system.flash_sdp=true \
    --system.compile=true \
    --logging.enabled=true 2>&1
