# ImageNet Experiments Tracker

## Overview
ImageNet-1K classification experiments using the Platonic Transformer architecture.
Comparing baseline (trivial group, translation-only equivariance via RoPE) against
C4-equivariant models.

## Environment
- **Cluster:** Snellius (SURF), H100 GPUs
- **Python:** 3.12.4 via uv venv
- **Key packages:** PyTorch 2.4.0+cu124, DALI 2.0, quack-kernels 0.3.3
- **CUDA:** 12.8 (module load)
- **W&B project:** Platonic-ImageNet

## Setup Notes
- Switched from conda (`plato-dali`) to uv venv (approved by David)
- 1-GPU runs are more efficient than 4-GPU for this model size (~80M params)
  - DDP communication overhead > parallelism gains for 20M param model
  - 4-GPU runs had NCCL timeout issues (possibly related to torch.compile)
- torch.compile was tested but reverted from the codebase (caused issues with multi-GPU, gains unclear on 1-GPU)
- batch_size=256 performs similarly to batch_size=1024 in wall-clock time per epoch
- DALI requires `DALI_NO_MMAP=1` on shared filesystems
- Lightning DDP: use ntasks=1 + unset SLURM vars, or ntasks=N + srun; we use the former

## Current Runs (2026-03-19)

| Job ID | Solid | hidden_dim | heads | layers | lr | rope_σ | ape_σ | freq_init | epochs | bs | GPUs | Status |
|--------|-------|-----------|-------|--------|------|--------|-------|-----------|--------|----|------|--------|
| 20933199 | trivial_2 | 768 | 12 | 12 | 8e-4 | 1.0 | 10.0 | random | 100 | 256 | 1×H100 | Running |
| 20933200 | cyclic_4 | 768 | 12 | 12 | 8e-4 | 1.0 | 10.0 | random | 100 | 256 | 1×H100 | Running |

Hyperparameters matched to the CIFAR-10 DeiT-III sweep configuration.

## Previous Runs (exploratory, 2026-03-18)

Several exploratory runs were done to establish the setup:
- Tested 4-GPU DDP vs 1-GPU: 1-GPU was faster for this model size
- Tested batch_size 256 vs 1024: similar wall-clock, 256 had better convergence
- Tested torch.compile: compiled but caused NCCL issues in multi-GPU; reverted
- Early runs used hidden_dim=384 (DeiT-Small), switched to 768 (DeiT-Base) to match CIFAR sweep

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/run_imagenet_trivial.sh` | 1-GPU trivial_2 baseline (current defaults) |
| `scripts/run_imagenet_c4.sh` | 1-GPU cyclic_4 equivariant (current defaults) |
| `scripts/run_imagenet_1gpu.sh` | 1-GPU generic (older, smaller model) |
| `scripts/run_imagenet_1gpu_c4.sh` | 1-GPU C4 generic (older, smaller model) |
| `scripts/run_imagenet_4gpu.sh` | 4-GPU (problematic, not recommended) |
| `scripts/run_dali_test.sh` | DALI smoke test (1 GPU, 3 batches) |
