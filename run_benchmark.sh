#!/bin/bash
#SBATCH --job-name=ntv3-benchmark
#SBATCH --output=slurm/log/ntv3-benchmark-%j.txt
#SBATCH --account=crescendo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4          
#SBATCH --time=0-12:00:00
#SBATCH --chdir=/e/project1/crescendo/reim1/genome-lm

set -euo pipefail

module purge
module use /e/project1/crescendo/hoffbauer1/easybuild/easybuild/jupiter/modules/all/Core
module load CUDA/12.8.0 GCC

export UV_CACHE_DIR=/e/project1/crescendo/reim1/uv-cache
export TRITON_HOME=/e/project1/crescendo/reim1/triton
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=WARN
mkdir -p "$UV_CACHE_DIR" "$TRITON_HOME"

cd /e/project1/crescendo/reim1/genome-lm
source .venv/bin/activate

torchrun --standalone --nproc_per_node=4 \
  /e/project1/crescendo/reim1/ntv3_benchmark/run_benchmark.py \
  --config /e/project1/crescendo/reim1/ntv3_benchmark/configs/nvt3_model_test.yaml \
  2> slurm/log/ntv3-benchmark-${SLURM_JOB_ID}.err