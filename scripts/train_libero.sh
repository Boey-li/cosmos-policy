# Set BASE_DATASETS_DIR to the directory containing the LIBERO-Cosmos-Policy dataset
export BASE_DATASETS_DIR=/coc/flash7/bli678/Projects/EgoVerse/external/cosmos-policy/LIBERO-Cosmos-Policy  # E.g., `/home/user/data/` if `LIBERO-Cosmos-Policy` is in this directory

# uv run --extra cu128 --group libero --python 3.10 \
#   torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_policy.scripts.train \
#   --config=cosmos_policy/config/config.py -- \
#   experiment="cosmos_predict2_2b_480p_libero" \
#   trainer.grad_accum_iter=8

python -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py \
  --dryrun