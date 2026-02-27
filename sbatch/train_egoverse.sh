#!/bin/bash
#SBATCH --job-name=cosmos_egoverse_train
#SBATCH --output=sbatch_logs/cosmos_egoverse_train.out
#SBATCH --error=sbatch_logs/cosmos_egoverse_train.err
#SBATCH --partition="rl2-lab"
#SBATCH --account="rl2-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node="l40s:1"
#SBATCH --qos="short"
#SBATCH --exclude="bishop"

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"

# Set BASE_DATASETS_DIR to the directory containing the EgoVerse dataset
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
source /coc/flash7/bli678/Projects/EgoVerse/emimic/bin/activate

# Run training
echo "Starting training..."
torchrun --nproc_per_node=1 -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py -- \
  experiment="cosmos_predict2_2b_480p_egoverse"

# Cleanup
echo "Training completed at: $(date +%Y-%m-%d_%H-%M-%S)"
echo "Cleaning up temporary directory: $TMPDIR"
rm -rf "$TMPDIR" 2>/dev/null || true

echo "Job finished at: $(date +%Y-%m-%d_%H-%M-%S)"

