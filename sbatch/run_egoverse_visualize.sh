#!/bin/bash
#SBATCH --job-name=cosmos_egoverse_visualize
#SBATCH --output=sbatch_logs/cosmos_egoverse_visualize.out
#SBATCH --error=sbatch_logs/cosmos_egoverse_visualize.err
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
mkdir -p sbatch_logs

# Print environment information
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "TMPDIR: $TMPDIR"
echo "BASE_DATASETS_DIR: $BASE_DATASETS_DIR"
echo "IMAGINAIRE_OUTPUT_ROOT: $IMAGINAIRE_OUTPUT_ROOT"

# Change to the project directory
cd /coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy
source /coc/flash7/bli678/Projects/EgoVerse/emimic/bin/activate

# Run visualization
echo "Starting visualization..."
python -m cosmos_policy.experiments.robot.egoverse.run_egoverse_eval_visualize \
  --config cosmos_predict2_2b_480p_egoverse \
  --ckpt_path /coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy/logs/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_eva/checkpoints/iter_000060000 \
  --config_file cosmos_policy/config/config.py \
  --chunk_size 25 \
  --num_denoising_steps 35 \
  --ar_future_prediction True \
  --num_denoising_steps_future_state 35 \
  --seed 195 \
  --deterministic True \
  --run_id_note visualize \
  --episode_indices 0 \
  --max_episodes 1 \
  --save_video_path ./visualizations \
  --local_log_dir cosmos_policy/experiments/robot/egoverse/logs/ \
  --video_fps 10 \

# Cleanup
echo "Visualization completed at: $(date +%Y-%m-%d_%H-%M-%S)"
echo "Cleaning up temporary directory: $TMPDIR"
rm -rf "$TMPDIR" 2>/dev/null || true

echo "Job finished at: $(date +%Y-%m-%d_%H-%M-%S)"

