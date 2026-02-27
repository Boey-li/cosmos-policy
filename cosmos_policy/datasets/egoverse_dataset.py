# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
EgoVerse robot tasks dataloader.

Wraps one or two S3RLDBDataset instances (demo + optional rollout).
__getitem__ is a single-sample version of
egomimic/algo/cosmos_policy.py::_robomimic_to_cosmos_policy_data.

Demo / rollout structure mirrors egomimic/hydra_configs/data/debug.yaml:
    demo    S3RLDBDataset: is_rollout=False  → rollout_data_mask=0
    rollout S3RLDBDataset: is_rollout=True   → rollout_data_mask=1

Video sequence per sample (use_proprio=True, use_values=False, D=num_duplicates_per_image):
  (1)  blank first input frame         1 repeat
  (2)  current proprio  (blank)        D repeats   ← only if use_proprio
  (3)  current left wrist image        D repeats   ← only if key present
  (4)  current right wrist image       D repeats   ← only if key present
  (5)  current primary image           D repeats
  (6)  action chunk     (blank)        D repeats
  (7)  future proprio   (blank)        D repeats   ← only if use_proprio
  (8)  future left wrist image         D repeats   ← only if key present
  (9)  future right wrist image        D repeats   ← only if key present
  (10) future primary image            D repeats
  (11) value            (blank)        D repeats   ← only if use_values

With use_proprio=True, use_values=False, both wrists present, D=4:
  state_t=10, chunk_duration = 1 + 9×4 = 37
"""

import copy
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

from egomimic.rldb.utils import S3RLDBDataset
from cosmos_policy.datasets.dataset_utils import preprocess_image, resize_images


class EgoVerseDataset(Dataset):
    """
    Single-sample wrapper around S3RLDBDataset that returns cosmos-policy format
    items.  __getitem__ exactly replicates egomimic's
    _robomimic_to_cosmos_policy_data without the batch loop.
    """

    def __init__(
        self,
        embodiment: str,
        mode: str = "train",
        bucket_name: str = "rldb",
        main_prefix: str = "processed_v2",
        temp_root: str = "/coc/flash7/scratch/egoverseS3Dataset/S3_rldb_data",
        cache_root: str = "/coc/flash7/scratch/.cache",
        # ── Demo data ─────────────────────────────────────────────────────────
        filters: dict = None,
        # ── Rollout data (mirrors debug.yaml rollout block) ───────────────────
        rollout_filters: dict = None,
        rollout_success: bool = True,
        p_world_model: float = 0.5,
        demonstration_sampling_prob: float = 1.0,
        return_value_function_returns: bool = False,
        gamma: float = 0.998,
        # ── General params ─────────────────────────────────────────────────────
        chunk_size: int = 25,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        use_proprio: bool = True,
        use_values: bool = True,
        num_duplicates_per_image: int = 4,
        local_files_only: bool = True,
        valid_ratio: float = 0.2,
        use_future: bool = True,
        # ── Camera / state key names ───────────────────────────────────────────
        # Key name pattern: egomimic detects role by substring:
        #   "front"       → primary camera
        #   "left_wrist"  → left wrist camera
        #   "right_wrist" → right wrist camera
        #   "future"      → future frame variant
        primary_camera_key: str = "observations.images.front_img_1",
        left_wrist_camera_key: str = "observations.images.left_wrist_img",
        right_wrist_camera_key: str = "observations.images.right_wrist_img",
        proprio_key: str = "observations.state.ee_pose",
        action_key: str = "actions_cartesian",
        debug: bool = False,
        **kwargs,
    ):
        """
        Args:
            filters: S3 filter dict for the demo S3RLDBDataset (is_rollout=False).
            rollout_filters: S3 filter dict for rollout S3RLDBDataset (is_rollout=True).
                             None → demo-only mode.
            rollout_success: Treat rollouts as successful (success=true in debug.yaml).
            p_world_model: Fraction of rollout samples for WM vs VF training.
            demonstration_sampling_prob: Fraction of epoch from demo dataset.
                                         1.0 = demo-only.  <1.0 mixes with rollouts.
            return_value_function_returns: Compute MC discounted returns when True.
            gamma: Discount factor for MC returns.
            chunk_size: Action chunk length; also passed as action_chunk to S3RLDBDataset.
            use_future: Enables future.* keys in S3RLDBDataset (must be True).
            use_proprio: Include proprio slot in video sequence.
            use_values: Include value slot in video sequence and supervise it.
        """
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.normalize_images = normalize_images
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio = use_proprio
        self.use_values = use_values
        self.num_duplicates_per_image = num_duplicates_per_image
        self.debug = debug
        self.demonstration_sampling_prob = demonstration_sampling_prob
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.p_world_model = p_world_model
        self.rollout_success = rollout_success

        # Camera / state key names
        self.primary_camera_key = primary_camera_key
        self.left_wrist_camera_key = left_wrist_camera_key
        self.right_wrist_camera_key = right_wrist_camera_key
        self.proprio_key = proprio_key
        self.action_key = action_key

        def _clean(f):
            if f is None:
                return {}
            try:
                from omegaconf import OmegaConf
                if OmegaConf.is_config(f):
                    f = OmegaConf.to_container(f, resolve=True)
            except ImportError:
                pass
            return dict(f)

        # ── Demo S3RLDBDataset ─────────────────────────────────────────────────
        self.demo_dataset = S3RLDBDataset(
            embodiment=embodiment,
            mode=mode,
            bucket_name=bucket_name,
            main_prefix=main_prefix,
            temp_root=temp_root,
            cache_root=cache_root,
            filters=_clean(filters),
            local_files_only=local_files_only,
            valid_ratio=valid_ratio,
            debug=debug,
            use_future=use_future,
            action_chunk=chunk_size,
            is_rollout=False,
            **kwargs,
        )
        self.s3_dataset = self.demo_dataset  # backward-compat alias

        # ── Rollout S3RLDBDataset (optional) ──────────────────────────────────
        self.rollout_dataset = None
        if rollout_filters is not None:
            self.rollout_dataset = S3RLDBDataset(
                embodiment=embodiment,
                mode=mode,
                bucket_name=bucket_name,
                main_prefix=main_prefix,
                temp_root=temp_root,
                cache_root=cache_root,
                filters=_clean(rollout_filters),
                local_files_only=local_files_only,
                valid_ratio=valid_ratio,
                debug=debug,
                use_future=use_future,
                action_chunk=chunk_size,
                is_rollout=True,
                success=rollout_success,
                p_world_model=p_world_model,
                **kwargs,
            )

        # ── T5 text embeddings ─────────────────────────────────────────────────
        self.t5_text_embeddings = None
        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as f:
                self.t5_text_embeddings = pickle.load(f)

        self.unique_commands = set()

        # ── Epoch length ───────────────────────────────────────────────────────
        self._demo_count    = len(self.demo_dataset)
        self._rollout_count = len(self.rollout_dataset) if self.rollout_dataset else 0
        self._epoch_length  = self._compute_epoch_length()

    # ------------------------------------------------------------------
    # Epoch / index helpers
    # ------------------------------------------------------------------

    def _compute_epoch_length(self) -> int:
        """Preserve demo/rollout ratio set by demonstration_sampling_prob."""
        if self._rollout_count == 0 or self.demonstration_sampling_prob >= 1.0:
            return self._demo_count

        p_demo    = max(1e-9, min(1.0, self.demonstration_sampling_prob))
        p_rollout = 1.0 - p_demo

        demo_scaled    = int(round(self._demo_count    / p_demo))
        rollout_scaled = int(round(self._rollout_count / p_rollout))
        epoch_length   = min(demo_scaled, rollout_scaled)

        self._adjusted_demo_count    = int(round(epoch_length * p_demo))
        self._adjusted_rollout_count = epoch_length - self._adjusted_demo_count
        return epoch_length

    def __len__(self):
        return 1 if self.debug else self._epoch_length

    def _resolve_idx(self, idx: int):
        """Return (sub_dataset, local_idx, is_rollout_sample)."""
        if self.rollout_dataset is None or not hasattr(self, "_adjusted_demo_count"):
            return self.demo_dataset, idx % max(1, self._demo_count), False
        if idx < self._adjusted_demo_count:
            return self.demo_dataset, idx % max(1, self._demo_count), False
        return self.rollout_dataset, (idx - self._adjusted_demo_count) % max(1, self._rollout_count), True

    # ------------------------------------------------------------------
    # Internal helpers — identical logic to _robomimic_to_cosmos_policy_data
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_numpy_uint8(raw) -> np.ndarray:
        """Convert a single image tensor/array to (H, W, C) uint8.
        Exact copy of egomimic's tensor_to_numpy_uint8 helper."""
        if isinstance(raw, torch.Tensor):
            if raw.dtype == torch.uint8:
                img_np = raw.cpu().numpy()
            elif raw.dtype in (torch.float32, torch.float64):
                img_np = raw.cpu().numpy()
                img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
            else:
                img_np = raw.cpu().numpy().astype(np.uint8)
        else:
            img_np = np.asarray(raw)
            if img_np.dtype != np.uint8:
                img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)

        # (C, H, W) → (H, W, C), ensure 3 channels
        if img_np.ndim == 3 and (img_np.shape[0] == 3 or img_np.shape[0] == 1):
            img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
        return img_np

    def _ensure_size(self, img: np.ndarray) -> np.ndarray:
        """Resize to (final_image_size, final_image_size) if needed."""
        if img.shape[0] != self.final_image_size or img.shape[1] != self.final_image_size:
            return resize_images(np.expand_dims(img, axis=0), self.final_image_size).squeeze(0)
        return img

    def _to_chunk(self, arr) -> np.ndarray:
        """Trim or pad a 2-D action/proprio array to (chunk_size, D) float32."""
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if len(arr) >= self.chunk_size:
            return arr[: self.chunk_size]
        last    = arr[-1] if len(arr) > 0 else np.zeros(arr.shape[-1], dtype=np.float32)
        padding = np.tile(last, (self.chunk_size - len(arr), 1))
        return np.concatenate([arr, padding], axis=0)

    def _get_image(self, item, key):
        """Return None if key absent, else (H,W,C) uint8 numpy array."""
        raw = item.get(key)
        if raw is None:
            return None
        return self._ensure_size(self._tensor_to_numpy_uint8(raw))

    # ------------------------------------------------------------------
    # __getitem__ — single-sample _robomimic_to_cosmos_policy_data
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        """
        Single-sample equivalent of egomimic's _robomimic_to_cosmos_policy_data.

        Reads one item from S3RLDBDataset (demo or rollout), extracts images /
        proprio / actions using the same key patterns as egomimic, builds the
        identical video sequence, and returns a flat dict ready for collation.
        """
        if self.debug:
            idx = 0

        sub_dataset, local_idx, is_rollout_sample = self._resolve_idx(idx)
        item = sub_dataset[local_idx]

        # ── Camera images ──────────────────────────────────────────────────────
        # egomimic detects roles by substring in key name:
        #   "front"       → primary   (current if no "future", future if "future")
        #   "left_wrist"  → left wrist
        #   "right_wrist" → right wrist
        primary_img        = self._get_image(item, self.primary_camera_key)
        left_wrist_img     = self._get_image(item, self.left_wrist_camera_key)
        right_wrist_img    = self._get_image(item, self.right_wrist_camera_key)
        future_primary_img = self._get_image(item, f"future.{self.primary_camera_key}")
        future_left_wrist_img  = self._get_image(item, f"future.{self.left_wrist_camera_key}")
        future_right_wrist_img = self._get_image(item, f"future.{self.right_wrist_camera_key}")

        # primary must always be valid (reference shape for blank frames)
        if primary_img is None:
            primary_img = np.zeros((self.final_image_size, self.final_image_size, 3), dtype=np.uint8)
        if future_primary_img is None:
            future_primary_img = np.zeros_like(primary_img)

        # ── Proprio ────────────────────────────────────────────────────────────
        # Concatenate all current proprio keys (keys without "future" in name)
        # mirrors: curr_proprio_list / torch.cat(curr_proprio_list, dim=-1)
        assert self.proprio_key in item, f"Key '{self.proprio_key}' missing."
        curr_proprio = np.asarray(
            item[self.proprio_key].cpu().numpy()
            if isinstance(item[self.proprio_key], torch.Tensor)
            else item[self.proprio_key],
            dtype=np.float32,
        )
        if curr_proprio.ndim > 1:
            curr_proprio = curr_proprio[0]

        future_proprio_raw = item.get(f"future.{self.proprio_key}")
        if future_proprio_raw is not None:
            future_proprio = np.asarray(
                future_proprio_raw.cpu().numpy()
                if isinstance(future_proprio_raw, torch.Tensor)
                else future_proprio_raw,
                dtype=np.float32,
            )
            if future_proprio.ndim > 1:
                future_proprio = future_proprio[0]
        else:
            future_proprio = curr_proprio.copy()

        # ── Actions ────────────────────────────────────────────────────────────
        # mirrors: curr_actions / future_actions (torch.cat along feature dim)
        assert self.action_key in item, f"Key '{self.action_key}' missing."
        curr_actions   = self._to_chunk(item[self.action_key])
        future_key     = f"future.{self.action_key}"
        future_actions = self._to_chunk(item[future_key]) if future_key in item else curr_actions.copy()

        # Slice to action_chunk exactly as egomimic does:
        #   curr_actions[batch_idx][:self.action_chunk, :]
        curr_action_chunk   = curr_actions[: self.chunk_size, :]
        future_action_chunk = future_actions[: self.chunk_size, :]

        # ── T5 text embeddings ─────────────────────────────────────────────────
        text_embeddings_path = '/coc/flash7/scratch/egowm/wmprocessedDataset/t5_embeddings.pkl'
        with open(text_embeddings_path, 'rb') as f:
            text_embeddings_dict = pickle.load(f)
        command = next(iter(text_embeddings_dict))
        t5_text_embeddings = text_embeddings_dict[command]  # (1, 512, 1024)
        t5_embedding = torch.squeeze(t5_text_embeddings)  # (512, 1024)

        # ── Build video sequence ───────────────────────────────────────────────
        # Single-sample version of the per-batch-item loop in egomimic.
        # Uses copy.deepcopy(primary_img) as ref shape, exactly as egomimic does.
        ref_image_for_shape = copy.deepcopy(primary_img)

        frames  = []   # list of (H, W, C) uint8 numpy arrays
        repeats = []   # repeat count per unique frame
        seg     = 0    # segment index counter (matches latent index in the model)

        # (1) blank first input frame
        frames.append(np.zeros_like(ref_image_for_shape)); repeats.append(1); seg += 1

        # (2) current proprio (blank image) — only if use_proprio
        if self.use_proprio:
            current_proprio_latent_idx = seg
            frames.append(np.zeros_like(ref_image_for_shape))
            repeats.append(self.num_duplicates_per_image)
            seg += 1
        else:
            current_proprio_latent_idx = -1

        # (3) current left wrist — only if image present
        if left_wrist_img is not None:
            current_wrist_image_latent_idx = seg
            frames.append(left_wrist_img)
            repeats.append(self.num_duplicates_per_image)
            seg += 1
        else:
            current_wrist_image_latent_idx = -1

        # (4) current right wrist — only if image present
        if right_wrist_img is not None:
            current_wrist_image2_latent_idx = seg
            frames.append(right_wrist_img)
            repeats.append(self.num_duplicates_per_image)
            seg += 1
        else:
            current_wrist_image2_latent_idx = -1

        # (5) current primary image
        current_image_latent_idx = seg
        frames.append(primary_img)
        repeats.append(self.num_duplicates_per_image)
        seg += 1

        # (6) action chunk (blank image)
        action_latent_idx = seg
        frames.append(np.zeros_like(ref_image_for_shape))
        repeats.append(self.num_duplicates_per_image)
        seg += 1

        # (7) future proprio (blank image) — only if use_proprio
        if self.use_proprio:
            future_proprio_latent_idx = seg
            frames.append(np.zeros_like(ref_image_for_shape))
            repeats.append(self.num_duplicates_per_image)
            seg += 1
        else:
            future_proprio_latent_idx = -1

        # (8) future left wrist — only if image present
        if future_left_wrist_img is not None:
            future_wrist_image_latent_idx = seg
            frames.append(future_left_wrist_img)
            repeats.append(self.num_duplicates_per_image)
            seg += 1
        else:
            future_wrist_image_latent_idx = -1

        # (9) future right wrist — only if image present
        if future_right_wrist_img is not None:
            future_wrist_image2_latent_idx = seg
            frames.append(future_right_wrist_img)
            repeats.append(self.num_duplicates_per_image)
            seg += 1
        else:
            future_wrist_image2_latent_idx = -1

        # (10) future primary image
        future_image_latent_idx = seg
        frames.append(future_primary_img)
        repeats.append(self.num_duplicates_per_image)
        seg += 1

        # (11) value (blank image) — only if use_values
        if self.use_values:
            value_latent_idx = seg
            frames.append(np.zeros_like(ref_image_for_shape))
            repeats.append(self.num_duplicates_per_image)
            seg += 1
        else:
            value_latent_idx = -1

        # Sanity: all non-(-1) latent indices must be within [0, num_segments)
        num_segments = len(frames)
        for name, val in (
            ("action_latent_idx",              action_latent_idx),
            ("value_latent_idx",               value_latent_idx),
            ("current_proprio_latent_idx",     current_proprio_latent_idx),
            ("current_wrist_image_latent_idx", current_wrist_image_latent_idx),
            ("current_wrist_image2_latent_idx",current_wrist_image2_latent_idx),
            ("current_image_latent_idx",       current_image_latent_idx),
            ("future_proprio_latent_idx",      future_proprio_latent_idx),
            ("future_wrist_image_latent_idx",  future_wrist_image_latent_idx),
            ("future_wrist_image2_latent_idx", future_wrist_image2_latent_idx),
            ("future_image_latent_idx",        future_image_latent_idx),
        ):
            if val != -1:
                assert 0 <= val < num_segments, f"{name}={val} out of range [0, {num_segments})"

        # Preprocess unique frames once (shared aug params across whole sequence)
        all_unique_images = np.stack(frames, axis=0)  # (num_segments, H, W, C)
        all_unique_images = preprocess_image(
            all_unique_images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )  # → (C, num_segments, H, W)

        # Expand by repeat counts along time dimension
        lengths    = torch.as_tensor(repeats, dtype=torch.long, device=all_unique_images.device)
        all_images = torch.repeat_interleave(all_unique_images, lengths, dim=1)  # (C, T_total, H, W)
        assert all_images.shape[1] == int(lengths.sum().item()), "T_total mismatch"

        # ── Demo / rollout masks ───────────────────────────────────────────────
        # S3RLDBDataset stamps these when is_rollout is set (Priority 1).
        rollout_data_mask          = int(item.get("rollout_data_mask",          int(is_rollout_sample)))
        rollout_data_success_mask  = int(item.get("rollout_data_success_mask",  int(is_rollout_sample and self.rollout_success)))
        world_model_sample_mask    = int(item.get("world_model_sample_mask",    0))
        value_function_sample_mask = int(item.get("value_function_sample_mask", 0))

        # global_rollout_idx: -1 for demos, random int [0, 10000) for rollouts
        # mirrors egomimic: torch.randint(0, 10000, (num_rollouts,))
        global_rollout_idx = -1 if rollout_data_mask == 0 else int(
            item.get("global_rollout_idx", np.random.randint(0, 10000))
        )

        # ── Value function returns (Monte-Carlo) ───────────────────────────────
        # Mirrors egomimic's _mc_return computation exactly.
        #   returns[t] = gamma^(T-1-t) * terminal_reward, rescaled to [-1, 1]
        #   terminal_reward = 1.0 for demos / success rollouts, 0.0 for failures
        def _mc_return(ep_len: int, future_t: int, terminal_reward: float) -> float:
            if ep_len <= 0 or future_t < 0:
                return -100.0
            t   = min(future_t, ep_len - 1)
            raw = (self.gamma ** (ep_len - 1 - t)) * terminal_reward
            if terminal_reward > 0:
                return float(2.0 * raw / terminal_reward - 1.0)  # rescale → [-1, 1]
            return -1.0

        if (
            self.return_value_function_returns
            and "frame_index"    in item
            and "episode_length" in item
        ):
            def _as_int(v):
                return int(v.item() if isinstance(v, torch.Tensor) else v)

            frame_idx = _as_int(item["frame_index"])
            ep_len    = _as_int(item["episode_length"])
            is_demo   = rollout_data_mask == 0
            is_succ   = rollout_data_success_mask == 1
            t_reward  = 1.0 if (is_demo or is_succ) else 0.0

            value_function_return      = _mc_return(ep_len, frame_idx +     self.chunk_size, t_reward)
            next_value_function_return = _mc_return(ep_len, frame_idx + 2 * self.chunk_size, t_reward)
        else:
            value_function_return      = -100.0
            next_value_function_return = -100.0

        # ── Return sample dict ─────────────────────────────────────────────────
        return {
            # Video sequence
            "video":                          all_images,                                   # (C, T_total, H, W)
            # Actions
            "actions":                        torch.from_numpy(curr_action_chunk).float(),  # (chunk_size, action_dim)
            "next_action_chunk":              torch.from_numpy(future_action_chunk).float(),# (chunk_size, action_dim)
            # Language
            "command":                        command,
            "t5_text_embeddings":             t5_embedding,                                 # (512, 1024)
            "t5_text_mask":                   torch.ones(512, dtype=torch.int64),
            # Metadata
            "fps":                            torch.tensor(30.0),                           # matches egomimic fps=30
            "padding_mask":                   torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size":                     self.final_image_size * torch.ones(4),
            "__key__":                        idx,
            # Proprio
            "proprio":                        torch.from_numpy(curr_proprio).float(),       # (proprio_dim,)
            "future_proprio":                 torch.from_numpy(future_proprio).float(),     # (proprio_dim,)
            # Value returns
            "value_function_return":          value_function_return,
            "next_value_function_return":     next_value_function_return,
            # Demo / rollout masks
            "rollout_data_mask":              rollout_data_mask,
            "rollout_data_success_mask":      rollout_data_success_mask,
            "world_model_sample_mask":        world_model_sample_mask,
            "value_function_sample_mask":     value_function_sample_mask,
            "global_rollout_idx":             global_rollout_idx,
            # Latent indices
            "action_latent_idx":              action_latent_idx,
            "value_latent_idx":               value_latent_idx,
            "current_proprio_latent_idx":     current_proprio_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_wrist_image2_latent_idx":current_wrist_image2_latent_idx,
            "current_image_latent_idx":       current_image_latent_idx,
            "future_proprio_latent_idx":      future_proprio_latent_idx,
            "future_wrist_image_latent_idx":  future_wrist_image_latent_idx,
            "future_wrist_image2_latent_idx": future_wrist_image2_latent_idx,
            "future_image_latent_idx":        future_image_latent_idx,
        }


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Demo-only (debug.yaml demo block only)
    ds = EgoVerseDataset(
        embodiment="eva_bimanual",
        mode="total",
        temp_root="/coc/flash7/scratch/egowm/wmprocessedDataset/",
        cache_root="/coc/flash7/mlin365/hf_cache",
        filters={"episode_hash": "2026-01-22-18-57-54-150000"},
        rollout_filters=None,
        chunk_size=25,
        t5_text_embeddings_path="/coc/flash7/scratch/egowm/wmprocessedDataset/t5_embeddings.pkl",
        use_proprio=True,
        use_values=True,
    )
    print(f"Dataset length: {len(ds)}")
    s = ds[0]
    print(f"video:   {s['video'].shape}")           # (C, T_total, H, W)
    print(f"actions: {s['actions'].shape}")         # (25, action_dim)
    print(f"rollout_data_mask:       {s['rollout_data_mask']}")
    print(f"action_latent_idx:       {s['action_latent_idx']}")
    print(f"future_image_latent_idx: {s['future_image_latent_idx']}")
    print(f"value_latent_idx:        {s['value_latent_idx']}")  # -1 since use_values=False
