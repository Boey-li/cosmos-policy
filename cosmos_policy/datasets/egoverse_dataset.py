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
EgoVerse robot tasks dataloader - wrapper around S3RLDBDataset to match ALOHADataset format.
"""

import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

from egomimic.rldb.utils import S3RLDBDataset
from cosmos_policy.datasets.dataset_common import get_action_chunk_with_padding
from cosmos_policy.datasets.dataset_utils import preprocess_image


class EgoVerseDataset(Dataset):
    """
    Wrapper around S3RLDBDataset that returns data in the same format as ALOHADataset.
    
    This allows S3RLDBDataset to be used as a drop-in replacement for ALOHADataset
    in training pipelines.
    """
    
    def __init__(
        self,
        embodiment: str,
        mode: str = "train",
        bucket_name: str = "rldb",
        main_prefix: str = "processed_v2",
        temp_root: str = "/coc/flash7/scratch/egoverseS3Dataset/S3_rldb_data",
        cache_root: str = "/coc/flash7/scratch/.cache",
        filters: dict = None,
        chunk_size: int = 25,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        use_proprio: bool = False,
        num_duplicates_per_image: int = 8,
        local_files_only: bool = True,
        valid_ratio: float = 0.2,
        # Camera key mappings (LeRobot dataset keys)
        primary_camera_key: str = "observations.images.cam_high",
        left_wrist_camera_key: str = "observations.images.left_wrist_img", 
        right_wrist_camera_key: str = "observations.images.right_wrist_img",
        proprio_key: str = "observations.state.ee_pose",
        action_key: str = "actions_cartesian",
        debug: bool = False,
        **kwargs,
    ):
        """
        Initialize EgoVerse dataset wrapper.
        
        Args:
            embodiment: Robot embodiment (e.g., "aria_bimanual", "eva_bimanual")
            mode: Dataset mode - "train", "valid", "total", or "percent"
            bucket_name: AWS S3 bucket name
            main_prefix: S3 prefix path to datasets
            temp_root: Local temp directory for downloaded datasets
            cache_root: HuggingFace cache directory
            filters: S3 filtering criteria (e.g., {"task": "fold_clothes", "lab": "rl2"})
            chunk_size: Action chunk size
            final_image_size: Target image size for resizing
            t5_text_embeddings_path: Path to precomputed T5 embeddings
            normalize_images: Whether to normalize images
            normalize_actions: Whether to normalize actions (handled by S3RLDBDataset)
            normalize_proprio: Whether to normalize proprio (handled by S3RLDBDataset)
            use_image_aug: Whether to apply image augmentations
            use_stronger_image_aug: Whether to apply stronger augmentations
            use_proprio: Whether to include proprioception
            num_duplicates_per_image: Temporal compression factor for tokenizer
            local_files_only: Whether to use only local files
            valid_ratio: Validation split ratio
            primary_camera_key: LeRobot key for primary camera
            left_wrist_camera_key: LeRobot key for left wrist camera
            right_wrist_camera_key: LeRobot key for right wrist camera
            proprio_key: LeRobot key for proprioception
            action_key: LeRobot key for actions
            debug: Debug mode (returns only first sample)
            **kwargs: Additional arguments passed to S3RLDBDataset
        """
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.debug = debug
        
        # Camera key mappings
        self.primary_camera_key = primary_camera_key
        self.left_wrist_camera_key = left_wrist_camera_key
        self.right_wrist_camera_key = right_wrist_camera_key
        self.proprio_key = proprio_key
        self.action_key = action_key
        
        # Initialize underlying S3RLDBDataset
        if filters is None:
            filters = {}
        
        self.s3_dataset = S3RLDBDataset(
            embodiment=embodiment,
            mode=mode,
            bucket_name=bucket_name,
            main_prefix=main_prefix,
            temp_root=temp_root,
            cache_root=cache_root,
            filters=filters,
            local_files_only=local_files_only,
            valid_ratio=valid_ratio,
            debug=debug,
            **kwargs,
        )
        
        # Load T5 text embeddings if provided
        self.t5_text_embeddings = None
        if t5_text_embeddings_path != "":
            with open(t5_text_embeddings_path, "rb") as f:
                self.t5_text_embeddings = pickle.load(f)
        
        # Cache unique commands from dataset
        self.unique_commands = set()
        
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        if self.debug:
            return 1
        return len(self.s3_dataset)
    
    def _extract_images_from_item(self, item):
        """Extract and convert images from LeRobot format to numpy arrays."""
        # Try to get images from different possible keys
        primary_image = None
        left_wrist_image = None
        right_wrist_image = None
        
        # Handle nested observation keys, need to ensure shape is (H, W, C) and dtype is uint8, sometimes it can be (C, H, W)
        if self.primary_camera_key in item:
            primary_image = item[self.primary_camera_key]
        
        if self.left_wrist_camera_key in item:
            left_wrist_image = item[self.left_wrist_camera_key]
            
        if self.right_wrist_camera_key in item:
            right_wrist_image = item[self.right_wrist_camera_key]
        
        # Convert tensors to numpy if needed
        def to_numpy(img):
            if img is None:
                return None
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            # Ensure shape is (H, W, C) - transpose if it's (C, H, W)
            if img.ndim == 3:
                if img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
            
            # Ensure uint8 format
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            return img
        
        primary_image = to_numpy(primary_image)
        left_wrist_image = to_numpy(left_wrist_image)
        right_wrist_image = to_numpy(right_wrist_image)
        
        # Create dummy images if any are missing
        if primary_image is not None:
            ref_image = primary_image
        elif left_wrist_image is not None:
            ref_image = left_wrist_image
        elif right_wrist_image is not None:
            ref_image = right_wrist_image
        else:
            # Create a default dummy image
            ref_image = np.zeros((self.final_image_size, self.final_image_size, 3), dtype=np.uint8)
        
        if primary_image is None:
            primary_image = np.zeros_like(ref_image)
        if left_wrist_image is None:
            left_wrist_image = np.zeros_like(ref_image)
        if right_wrist_image is None:
            right_wrist_image = np.zeros_like(ref_image)
            
        return primary_image, left_wrist_image, right_wrist_image
    
    def _get_episode_data(self, idx):
        """Get the episode index and relative step index for a given global index."""
        # S3RLDBDataset already provides episode information
        item = self.s3_dataset[idx]
        
        # Get episode information from the underlying dataset
        # Note: This assumes episode-related information is available
        episode_idx = int(item.get("episode_index", 0))
        
        # Get the actual episode data from the underlying HF dataset
        hf_dataset = self.s3_dataset.hf_dataset
        
        return item, episode_idx, hf_dataset
    
    def __getitem__(self, idx):
        """
        Fetches data sample in ALOHADataset format.
        
        Returns:
            dict: Data sample with keys matching ALOHADataset output format:
                - video: preprocessed images (n_segments, T, C, H, W)
                - command: text instruction
                - actions: action chunk
                - t5_text_embeddings: text embedding
                - t5_text_mask: text embedding mask
                - fps: frames per second
                - padding_mask: padding mask
                - image_size: image size tensor
                - proprio: current proprioception
                - future_proprio: future proprioception
                - __key__: sample index
                - Various latent indices for different modalities
        """
        if self.debug:
            idx = 0
        
        # Get current timestep data from S3RLDBDataset
        current_item = self.s3_dataset[idx]
        
        # Extract images
        primary_current, left_current, right_current = self._extract_images_from_item(current_item)
        
        # Extract proprioception
        assert self.proprio_key in current_item, f"Expected key '{self.proprio_key}' in current_item"
        proprio = current_item[self.proprio_key]
        if isinstance(proprio, torch.Tensor):
            proprio = proprio.cpu().numpy()
        
        # Extract action chunk (already a chunk, just need to trim/pad)
        assert self.action_key in current_item, f"Expected key '{self.action_key}' in current_item"
        action_chunk = current_item[self.action_key]
        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.cpu().numpy()
        
        # Trim or pad action chunk to match desired chunk_size
        current_chunk_len = len(action_chunk)
        if current_chunk_len >= self.chunk_size:
            # Trim to chunk_size
            action_chunk = action_chunk[:self.chunk_size]
        else:
            # Pad with the last action repeated
            last_action = action_chunk[-1] if len(action_chunk) > 0 else np.zeros(14, dtype=np.float32)
            padding = np.tile(last_action, (self.chunk_size - current_chunk_len, 1))
            action_chunk = np.concatenate([action_chunk, padding], axis=0)
        
        # Extract future proprio (with "future." prefix)
        future_proprio_key = f"future.{self.proprio_key}"
        if future_proprio_key in current_item:
            future_proprio = current_item[future_proprio_key]
            if isinstance(future_proprio, torch.Tensor):
                future_proprio = future_proprio.cpu().numpy()
        else:
            # Fallback to current proprio if future not available
            future_proprio = proprio.copy()
        
        # Extract future images (with "future." prefix)
        future_primary_key = f"future.{self.primary_camera_key}"
        future_left_wrist_key = f"future.{self.left_wrist_camera_key}"
        future_right_wrist_key = f"future.{self.right_wrist_camera_key}"
        
        # Create a future item dict with the future keys
        future_item = {
            self.primary_camera_key: current_item.get(future_primary_key),
            self.left_wrist_camera_key: current_item.get(future_left_wrist_key),
            self.right_wrist_camera_key: current_item.get(future_right_wrist_key),
        }
        
        # Get future images
        primary_future, left_future, right_future = self._extract_images_from_item(future_item)
        
        # Get command/instruction
        command = current_item.get("annotations", "")
        if not command:
            # command = "robot task"  # Default fallback
            command = "pick up cup with one arm, hand it over to the other arm and place it on the saucer"
        self.unique_commands.add(command)
        
        # Build video tensor with proper structure (like ALOHA)
        # Structure: [blank, current_proprio, left_wrist, right_wrist, primary, action, 
        #             future_proprio, future_left_wrist, future_right_wrist, future_primary]
        frames = []
        repeats = []
        
        # Add blank first input frame
        blank_first_input_frame = np.zeros_like(primary_current)
        frames.append(blank_first_input_frame)
        repeats.append(1)
        
        segment_idx = 1
        
        # Add current proprio (as blank image)
        if self.use_proprio:
            blank_proprio_image = np.zeros_like(primary_current)
            current_proprio_latent_idx = segment_idx
            frames.append(blank_proprio_image)
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
        else:
            current_proprio_latent_idx = -1
        
        # Add current left wrist image
        current_wrist_image_latent_idx = segment_idx
        frames.append(left_current)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1
        
        # Add current right wrist image
        current_wrist_image2_latent_idx = segment_idx
        frames.append(right_current)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1
        
        # Add current primary image
        current_image_latent_idx = segment_idx
        frames.append(primary_current)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1
        
        # Add blank image for action chunk
        blank_action_image = np.zeros_like(primary_current)
        action_latent_idx = segment_idx
        frames.append(blank_action_image)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1
        
        # Add future proprio (as blank image)
        if self.use_proprio:
            blank_future_proprio_image = np.zeros_like(primary_current)
            future_proprio_latent_idx = segment_idx
            frames.append(blank_future_proprio_image)
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
        else:
            future_proprio_latent_idx = -1
        
        # Add future left wrist image
        future_wrist_image_latent_idx = segment_idx
        frames.append(left_future)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1
        
        # Add future right wrist image
        future_wrist_image2_latent_idx = segment_idx
        frames.append(right_future)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1
        
        # Add future primary image
        future_image_latent_idx = segment_idx
        frames.append(primary_future)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1
        
        # Concatenate and preprocess all unique frames
        # print([f.shape for f in frames])  # Debug: print shapes of all frames
        all_unique_images = np.stack(frames, axis=0)  # (num_segments, H, W, C)
        all_unique_images = preprocess_image(
            all_unique_images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        
        # Expand by repeat counts
        lengths = torch.as_tensor(repeats, dtype=torch.long, device=all_unique_images.device)
        all_images = torch.repeat_interleave(all_unique_images, lengths, dim=1)  # (n_segments, T, C, H, W)
        
        # Get T5 text embeddings
        assert self.t5_text_embeddings is not None and command in self.t5_text_embeddings, f"Command '{command}' not in T5 embeddings {self.t5_text_embeddings.keys()}"
        t5_embedding = torch.squeeze(self.t5_text_embeddings[command])
       
        # Convert actions to tensor
        action_chunk = torch.from_numpy(action_chunk).float()
        
        # Return in ALOHADataset format
        return {
            "video": all_images,
            "command": command,
            "actions": action_chunk,
            "t5_text_embeddings": t5_embedding,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio,
            "future_proprio": future_proprio,
            "__key__": idx,
            "value_function_return": -100.0,  # Placeholder
            "next_action_chunk": action_chunk,  # Simplified: same as current
            "next_value_function_return": -100.0,  # Placeholder
            "rollout_data_mask": 0,  # Assume all are demos
            "rollout_data_success_mask": 1,  # Assume all successful
            "world_model_sample_mask": 1,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": -1,  # Not used
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_wrist_image2_latent_idx": current_wrist_image2_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_wrist_image2_latent_idx": future_wrist_image2_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
        }


if __name__ == "__main__":
    # Example usage
    dataset = EgoVerseDataset(
        embodiment="eva_bimanual",
        mode="valid",
        temp_root='/coc/flash7/scratch/egowm/wmprocessedDataset/',
        cache_root='/coc/flash7/mlin365/hf_cache',
        filters={"episode_hash":"2026-01-22-18-57-54-150000"},
        # filters={"lab": "rl2", "task": "put cup on saucer indomain"},
        chunk_size=4,
        final_image_size=224,
        t5_text_embeddings_path='/coc/flash7/scratch/egowm/wmprocessedDataset/t5_embeddings.pkl'
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Video shape: {sample['video'].shape}")
    print(f"Actions shape: {sample['actions'].shape}")
    print(f"Command: {sample['command']}")
