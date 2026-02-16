#!/bin/bash
#SBATCH --job-name=cosmos_libero_train
#SBATCH --output=sbatch_logs/cosmos_libero_train.out
#SBATCH --error=sbatch_logs/cosmos_libero_train.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Set BASE_DATASETS_DIR to the directory containing the LIBERO-Cosmos-Policy dataset
export BASE_DATASETS_DIR="/coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy"

# Use local filesystem for TMPDIR to avoid NFS cleanup issues
# /dev/shm is shared memory (fast, local) - fallback to /tmp if not available
export TMPDIR="${TMPDIR:-/dev/shm}"
if [ ! -d "$TMPDIR" ] || [ ! -w "$TMPDIR" ]; then
    export TMPDIR="/tmp"
fi
mkdir -p "$TMPDIR/cosmos_policy_$$"
export TMPDIR="$TMPDIR/cosmos_policy_$$"

export IMAGINAIRE_OUTPUT_ROOT="/coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy/logs"

# Create logs directory if it doesn't exist
mkdir -p logs

# Print environment information
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TMPDIR: $TMPDIR"
echo "BASE_DATASETS_DIR: $BASE_DATASETS_DIR"
echo "IMAGINAIRE_OUTPUT_ROOT: $IMAGINAIRE_OUTPUT_ROOT"

# Change to the project directory
cd /coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy
source /coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy/.venv/bin/activate

# Run training
echo "Starting training..."
uv run --extra cu128 --group libero --python 3.10 \
  torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_libero_one_demo_one_episode" \
  trainer.grad_accum_iter=1

# Cleanup
echo "Training completed at: $(date)"
echo "Cleaning up temporary directory: $TMPDIR"
rm -rf "$TMPDIR" 2>/dev/null || true

echo "Job finished at: $(date)"

