import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
from daart.transformer_loader_chunks import extract_patch_tokens_chunk_gpu
from torchvision import transforms

class OpenCVVideoDataset(Dataset):
    """
    Dataset for loading video frames using OpenCV and extracting transformer features.
    This implementation bypasses DALI's limitations with specific frame access.
    """
    def __init__(
            self,
            expt_ids,
            video_dir,
            transformer_config,
            transformer_ckpt,
            sequence_length=64,
            labels_dir=None,
            data_dir=None,
            device="cuda"
    ):
        """
        Initialize the OpenCV-based video dataset.
        
        Args:
            expt_ids (list): List of experiment IDs (video names without extension)
            video_dir (str): Directory containing the videos
            transformer_config (str): Path to transformer config file
            transformer_ckpt (str): Path to transformer checkpoint file
            sequence_length (int): Number of frames per sequence
            labels_dir (str, optional): Directory containing label files
            data_dir (str, optional): Alternative directory for label files
            device (str): Device to use for processing ('cuda' or 'cpu')
        """
        self.expt_ids = expt_ids
        self.video_dir = video_dir
        self.transformer_config = transformer_config
        self.transformer_ckpt = transformer_ckpt
        self.sequence_length = sequence_length
        self.labels_dir = labels_dir
        self.data_dir = data_dir if data_dir is not None else '.'
        self.device = device
        
        # Initialize feature size (will be set on first data load)
        self.input_size = None
        self.feature_names = None
        self.label_names = None  # Initialize label_names before _load_labels is called
        self.label_cache = {}    # Initialize label_cache
        
        # Build video index
        self._build_video_index()
        
        # Load labels
        self._load_labels()

    def _build_video_index(self):
        """Build index of all video frames"""
        self.video_info = {}  # mapping from eid to total frames
        self.index_map = []   # list of (eid, start_frame, end_frame, total_frames, idx)
        
        print("Building video index...")
        for eid in self.expt_ids:
            video_path = os.path.join(self.video_dir, f"{eid}.mp4")
            if os.path.exists(video_path):
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                self.video_info[eid] = {
                    'total_frames': total_frames,
                    'path': video_path
                }
                
                # Create windows based on sequence_length
                n_windows = (total_frames + self.sequence_length - 1) // self.sequence_length
                for w in range(n_windows):
                    start_frame = w * self.sequence_length
                    end_frame = min(start_frame + self.sequence_length, total_frames)
                    self.index_map.append((eid, start_frame, end_frame, total_frames, len(self.index_map)))
            else:
                print(f"Warning: Video file {video_path} not found")
        
        print(f"Indexed {len(self.index_map)} sequences from {len(self.video_info)} videos")
    
    def _load_labels(self):
        """Preload all label files for faster access"""
        self.label_cache = {}
        
        # Determine the device to load tensors to
        device = self.device  # Use the class device attribute
        
        print(f"Loading labels to device: {device}...")
        for eid in self.expt_ids:
            # Try labels_dir first, then data_dir/label-hands
            label_path = None
            if self.labels_dir is not None:
                label_path = os.path.join(self.labels_dir, f"{eid}_labels.csv")
            
            if label_path is None or not os.path.exists(label_path):
                label_path = os.path.join(self.data_dir, 'labels-hand', f"{eid}_labels.csv")
            
            if os.path.exists(label_path):
                df = pd.read_csv(label_path)
                
                # Store column names as label names if not already set
                if self.label_names is None:
                    # Skip the first column (frame index)
                    self.label_names = list(df.columns[1:])
                
                # Convert one-hot to indices
                labels = np.argmax(df.values[:, 1:], axis=1)
                
                # Convert to tensor and move directly to the target device
                self.label_cache[eid] = torch.tensor(labels, dtype=torch.long, device=device)
                
                print(f"Loaded labels for {eid}: {len(labels)} frames, device: {self.label_cache[eid].device}")
            else:
                print(f"Warning: Label file for {eid} not found")
    
    def __len__(self):
        """Return the number of sequences in the dataset"""
        return len(self.index_map)
    
    def __getitem__(self, idx):
        """Get a specific sequence by index"""
        eid, start_frame, end_frame, total_frames, _ = self.index_map[idx]
        return self.get_specific_frames(eid, start_frame, end_frame, idx)
    
    def get_specific_frames(self, eid, start_frame, end_frame=None, idx=None):
        """
        Get a specific video segment identified by experiment ID and frame range.
        
        Args:
            eid (str): Experiment ID
            start_frame (int): Starting frame index
            end_frame (int, optional): Ending frame index (exclusive)
                If None, will use start_frame + sequence_length
            idx (int, optional): Index to return as batch_idx
                
        Returns:
            dict: Dictionary containing transformer features and labels
        """
        if eid not in self.video_info:
            raise ValueError(f"Experiment ID {eid} not found in dataset")
        
        total_frames = self.video_info[eid]['total_frames']
        
        # Set end_frame if not provided
        if end_frame is None:
            end_frame = min(start_frame + self.sequence_length, total_frames)
        else:
            end_frame = min(end_frame, total_frames)
        
        # Check frame range
        if start_frame < 0 or start_frame >= total_frames:
            raise ValueError(f"Start frame {start_frame} out of range (0-{total_frames-1})")
        if end_frame <= start_frame or end_frame > total_frames:
            raise ValueError(f"End frame {end_frame} out of range ({start_frame+1}-{total_frames})")
        
        # Calculate number of frames to load
        frame_count = end_frame - start_frame
        
        # Load frames using OpenCV
        video_path = self.video_info[eid]['path']
        cap = cv2.VideoCapture(video_path)
        
        # Set the position to start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frame_count frames
        frames = []
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB (OpenCV loads as BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        # Release the video capture
        cap.release()
        
        # Convert to numpy array
        frames = np.array(frames)
        
        # Apply image transforms to frames before feature extraction
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((224, 224)),  # Resize to the input size expected by the model
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet norms
        ])
        
        # Apply transforms to each frame
        transformed_frames = []
        for frame in frames:
            transformed_frame = transform(frame)  # Shape: [C, H, W]
            transformed_frames.append(transformed_frame)
        
        # Stack the transformed frames
        transformed_frames = torch.stack(transformed_frames)  # Shape: [T, C, H, W]
        
        # Move to device if needed
        if self.device == 'cuda':
            transformed_frames = transformed_frames.to(self.device)
        
        # Process frames with transformer
        transformer_features = extract_patch_tokens_chunk_gpu(
            transformed_frames,
            self.transformer_config,
            self.transformer_ckpt,
            self.sequence_length,
            device=self.device
        )
        
        # Convert to torch tensor and transpose from (T, C) to (C, T)
        transformer_features = transformer_features.float() #torch.from_numpy(transformer_features.T).float()
        
        # Set input size if not already set
        if self.input_size is None:
            self.input_size = transformer_features.shape[0]
            self.feature_names = [f"tok{i}" for i in range(self.input_size)]
        
        # Get labels if available
        labels = None
        if eid in self.label_cache:
            # Get labels for the specific frame range
            labels = self.label_cache[eid][start_frame:end_frame]
            
            # If window shorter than sequence_length, pad to sequence_length
            if frame_count < self.sequence_length:
                pad_len = self.sequence_length - frame_count
                transformer_features = F.pad(transformer_features, (0, pad_len))
                labels = F.pad(labels, (0, pad_len), value=0)
        
        # Create result dictionary
        result = OrderedDict()
        result['transformer'] = transformer_features
        
        if labels is not None:
            result['labels_strong'] = labels
        
        # Add metadata
        result['batch_idx'] = idx if idx is not None else 0
        result['eid'] = eid
        result['start_frame'] = start_frame
        result['end_frame'] = end_frame
        
        return result

    def get_batch(self, indices):
        """
        Get a batch of samples for the specified indices.
        
        Args:
            indices (list): List of dataset indices to include in batch
            
        Returns:
            dict: Batch dictionary with stacked tensors
        """
        # Initialize batch containers
        batch = OrderedDict()
        batch['transformer'] = []
        batch['batch_idx'] = indices
        batch['eid'] = []
        batch['start_frame'] = []
        batch['end_frame'] = []
        
        has_labels = False
        
        # Load each sample
        for i, idx in enumerate(indices):
            sample = self[idx]
            
            # Check if we have labels
            if 'labels_strong' in sample and not has_labels:
                batch['labels_strong'] = []
                has_labels = True
            
            # Add to batch
            for k in batch:
                if k != 'batch_idx':  # batch_idx already set
                    batch[k].append(sample[k])
            
            # Add labels if available
            if has_labels and 'labels_strong' in sample:
                batch['labels_strong'].append(sample['labels_strong'])
        
        # Stack tensors
        batch['transformer'] = torch.stack(batch['transformer'])
        
        if has_labels:
            batch['labels_strong'] = torch.stack(batch['labels_strong'])
        
        return batch


class OpenCVDataGenerator:
    """
    Data generator for the OpenCVVideoDataset.
    Provides training, validation, and testing splits with batch handling.
    """
    def __init__(
            self,
            dataset,
            batch_size=8,
            shuffle=True,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
            rng_seed=0,
            num_workers=4,
            pin_memory=True
    ):
        """
        Initialize a data generator.
        
        Args:
            dataset: The OpenCVVideoDataset to use
            batch_size (int): Batch size for training
            shuffle (bool): Whether to shuffle the data
            train_frac (float): Fraction of data to use for training
            val_frac (float): Fraction of data to use for validation
            test_frac (float): Fraction of data to use for testing
            rng_seed (int): Random seed for reproducibility
            num_workers (int): Number of worker processes for DataLoader
            pin_memory (bool): Whether to pin memory for faster GPU transfer
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.rng_seed = rng_seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Add compatibility attributes to match the original DataGenerator class
        # n_datasets should reflect the number of experiment IDs
        self.n_datasets = len(dataset.expt_ids) if hasattr(dataset, 'expt_ids') and dataset.expt_ids else 1
        self.datasets = [dataset]  # Wrap the dataset in a list to match old interface
        self.input_size = dataset.input_size
        self.feature_names = dataset.feature_names
        self.label_names = dataset.label_names
        
        # Create data splits
        self._create_splits()
        
        # Create dataloaders
        self._create_dataloaders()
    
    def _create_splits(self):
        """Create train/val/test splits of the dataset"""
        # Get dataset size
        n_samples = len(self.dataset)
        
        # Set random seed for reproducibility
        np.random.seed(self.rng_seed)
        
        # Create random permutation of indices
        indices = np.random.permutation(n_samples)
        
        # Calculate split sizes
        n_train = int(n_samples * self.train_frac)
        n_val = int(n_samples * self.val_frac)
        
        # Create split indices
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:n_train + n_val]
        self.test_indices = indices[n_train + n_val:]
        
        print(f"Split dataset into {len(self.train_indices)} train, "
              f"{len(self.val_indices)} validation, "
              f"{len(self.test_indices)} test samples")
    
    def _create_dataloaders(self):
        """Create DataLoader objects for each split"""
        from torch.utils.data import Subset, DataLoader, SubsetRandomSampler
        
        # Create samplers for each split
        train_sampler = SubsetRandomSampler(self.train_indices) if self.shuffle else None
        
        # Set num_workers to 0 to avoid multiprocessing issues
        # This is a fix for the "daemonic processes are not allowed to have children" error
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None and self.shuffle),
            num_workers=0,  # Set to 0 to avoid multiprocessing errors
            #pin_memory=self.pin_memory
        )
        
        self.val_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.val_indices),
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing errors
            pin_memory=self.pin_memory
        )
        
        self.test_loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=SubsetRandomSampler(self.test_indices),
            shuffle=False,
            num_workers=0,  # Set to 0 to avoid multiprocessing errors
            pin_memory=self.pin_memory
        )
        
        # Add compatibility with the original DataGenerator interface
        self.n_tot_batches = {
            'train': len(self.train_loader),
            'val': len(self.val_loader),
            'test': len(self.test_loader)
        }
    
    def __iter__(self):
        """Iterator for training batches (default iteration is training)"""
        return iter(self.train_loader)
    
    def __len__(self):
        """Return the number of batches in the training set"""
        return len(self.train_loader)
    
    def train_batches(self):
        """Generator for training batches"""
        for batch in self.train_loader:
            yield batch
    
    def val_batches(self):
        """Generator for validation batches"""
        for batch in self.val_loader:
            yield batch
    
    def test_batches(self):
        """Generator for test batches"""
        for batch in self.test_loader:
            yield batch
    
    # Add compatibility methods with original DataGenerator
    def reset_iterators(self, dtype):
        """Reset iterators so that all data is available."""
        if hasattr(self, '_iterators'):
            if dtype == 'all':
                for dt in ['train', 'val', 'test']:
                    self._iterators[dt] = iter(getattr(self, f"{dt}_loader"))
            else:
                self._iterators[dtype] = iter(getattr(self, f"{dtype}_loader"))
    
    def next_batch(self, dtype, transforms=None):
        """Return next batch of data using the specified data type."""
        if dtype == 'train':
            loader = self.train_loader
        elif dtype == 'val':
            loader = self.val_loader
        elif dtype == 'test':
            loader = self.test_loader
        else:
            raise ValueError(f"Unknown data type: {dtype}")
        
        # Instead of creating a new iterator each time, use class-level iterators
        if not hasattr(self, '_iterators'):
            self._iterators = {
                'train': iter(self.train_loader),
                'val': iter(self.val_loader),
                'test': iter(self.test_loader)
            }
        
        try:
            batch = next(self._iterators[dtype])
            batch['markers'] = batch['transformer']
            return batch, [0]  # Return batch and a list with a single dataset index
        except StopIteration:
            # Reset the iterator and try again
            self._iterators[dtype] = iter(loader)
            try:
                batch = next(self._iterators[dtype])
                batch['markers'] = batch['transformer']
                return batch, [0]
            except StopIteration:
                # If we still get StopIteration, the loader must be empty
                return False, False
    
    def get_specific_batch(self, eids, start_frames, end_frames=None):
        """
        Get a batch containing specific frames from specific videos.
        
        Args:
            eids (list): List of experiment IDs
            start_frames (list): List of start frames corresponding to each eid
            end_frames (list, optional): List of end frames corresponding to each eid
            
        Returns:
            dict: Batch dictionary with stacked tensors
        """
        if len(eids) != len(start_frames):
            raise ValueError("eids and start_frames must have the same length")
        
        if end_frames is not None and len(eids) != len(end_frames):
            raise ValueError("eids and end_frames must have the same length")
        
        samples = []
        for i, (eid, start_frame) in enumerate(zip(eids, start_frames)):
            end_frame = None if end_frames is None else end_frames[i]
            sample = self.dataset.get_specific_frames(eid, start_frame, end_frame, i)
            samples.append(sample)
        
        # Initialize batch containers
        batch = OrderedDict()
        batch['transformer'] = []
        batch['batch_idx'] = list(range(len(samples)))
        batch['eid'] = []
        batch['start_frame'] = []
        batch['end_frame'] = []
        
        has_labels = False
        
        # Gather samples into a batch
        for i, sample in enumerate(samples):
            # Check if we have labels
            if 'labels_strong' in sample and not has_labels:
                batch['labels_strong'] = []
                has_labels = True
            
            # Add to batch
            for k in batch:
                if k != 'batch_idx':  # batch_idx already set
                    batch[k].append(sample[k])
            
            # Add labels if available
            if has_labels and 'labels_strong' in sample:
                batch['labels_strong'].append(sample['labels_strong'])
        
        # Stack tensors
        batch['transformer'] = torch.stack(batch['transformer'])
        
        if has_labels:
            batch['labels_strong'] = torch.stack(batch['labels_strong'])
        
        return batch
    
    def get_labeled_frames_batch(self, labels_dict):
        """
        Get a batch containing frames with specific labels.
        
        Args:
            labels_dict (dict): Dictionary mapping experiment IDs to frame indices
            
        Returns:
            dict: Batch dictionary with stacked tensors
        """
        eids = []
        start_frames = []
        
        for eid, frames in labels_dict.items():
            for frame in frames:
                eids.append(eid)
                start_frames.append(frame)
        
        return self.get_specific_batch(eids, start_frames)
    
    def get_input_size(self):
        """Get the input size (transformer feature dimension)"""
        return self.dataset.input_size
    
    def get_output_size(self):
        """Get the output size (number of label classes)"""
        return len(self.dataset.label_names) if self.dataset.label_names else 0
    
    def get_feature_names(self):
        """Get the feature names"""
        return self.dataset.feature_names
    
    def get_label_names(self):
        """Get the label names"""
        return self.dataset.label_names
    
    # Additional method needed for compatibility
    def count_class_examples(self):
        """Count the number of examples for each class in the dataset"""
        import numpy as np
        
        if not hasattr(self.dataset, 'label_names') or self.dataset.label_names is None:
            return None
        
        totals = np.zeros(len(self.dataset.label_names), dtype=int)
        
        # This is a simplified version - in the actual implementation
        # you might need to iterate through your dataset to count labels
        # For example:
        for idx in self.train_indices:
            sample = self.dataset[idx]
            if 'labels_strong' in sample:
                labels = sample['labels_strong'].cpu().numpy()
                for label in labels:
                    if 0 <= label < len(totals):
                        totals[label] += 1
        
        return totals