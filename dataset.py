#!/usr/bin/env python3
"""
dataset.py - Dataset classes for anomaly detection datasets
Dedicated classes tailored to the characteristics of each dataset
"""

import os
import numpy as np
from PIL import Image
from glob import glob
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


class BaseAnomalyDataset(Dataset, ABC):
    """Base anomaly detection dataset class"""
    
    def __init__(self, dataset_dir, split='all'):
        self.dataset_dir = dataset_dir
        self.split = split
        self.image_paths = []
        self.label_paths = []
        self.dataset_name = self.__class__.__name__
        
        # Load data
        self._load_data()
        
        print(f"Loaded {self.dataset_name}: {len(self.image_paths)} samples")
        if len(self.image_paths) > 0:
            print(f"  Sample files: {os.path.basename(self.image_paths[0])} -> {os.path.basename(self.label_paths[0])}")
    
    @abstractmethod
    def _load_data(self):
        """Implement data loading for each dataset"""
        pass
    
    @abstractmethod
    def _process_mask(self, mask):
        """Implement mask processing for each dataset"""
        pass
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Load and process mask
        mask = np.array(Image.open(label_path))
        binary_mask = self._process_mask(mask)
        
        return {
            'image': image,
            'mask': binary_mask,
            'image_path': image_path,
            'label_path': label_path,
            'image_name': os.path.basename(image_path)
        }
    
    def get_sample_info(self, idx):
        """Return sample information"""
        sample = self[idx]
        return {
            'index': idx,
            'image_name': sample['image_name'],
            'image_size': sample['image'].size,
            'mask_shape': sample['mask'].shape,
            'positive_pixels': int(np.sum(sample['mask'])),
            'positive_ratio': float(np.sum(sample['mask']) / sample['mask'].size)
        }


class RoadAnomalyDataset(BaseAnomalyDataset):
    """Road Anomaly dataset
    - Mask values: 0 (background), 1 (anomaly)
    - File format: .jpg -> .png
    """
    
    def _load_data(self):
        original_dir = os.path.join(self.dataset_dir, 'original')
        label_dir = os.path.join(self.dataset_dir, 'labels')
        
        if not os.path.exists(original_dir) or not os.path.exists(label_dir):
            print(f"Warning: Directory not found - {original_dir} or {label_dir}")
            return
        
        # Find image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(original_dir, ext)))
        
        # Find matching labels
        for image_path in sorted(image_files):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, f"{base_name}.png")
            
            if os.path.exists(label_path):
                self.image_paths.append(image_path)
                self.label_paths.append(label_path)
            else:
                print(f"Warning: No label found for {base_name}")
    
    def _process_mask(self, mask):
        """Road Anomaly: treat all non-zero values as positive"""
        # Check mask dimensions
        if len(mask.shape) == 3:
            # If RGB mask - convert to grayscale by averaging
            mask = mask.mean(axis=2)
        
        # Treat non-zero values as positive (0-1 range mask)
        binary_mask = (mask > 0).astype(bool)
        
        return binary_mask


class FishyscapesDataset(BaseAnomalyDataset):
    """Fishyscapes dataset (LostAndFound, Static)
    - Mask values: 0 (background), 1 (OOD object), 255 (known object)
    - For OOD detection, only pixels with value 1 are treated as positive
    - File format: .png -> .png
    """
    
    def _load_data(self):
        original_dir = os.path.join(self.dataset_dir, 'original')
        label_dir = os.path.join(self.dataset_dir, 'labels')
        
        if not os.path.exists(original_dir) or not os.path.exists(label_dir):
            print(f"Warning: Directory not found - {original_dir} or {label_dir}")
            return
        
        # Find PNG image files
        image_files = glob(os.path.join(original_dir, '*.png'))
        
        # Find matching labels
        for image_path in sorted(image_files):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(label_dir, f"{base_name}.png")
            
            if os.path.exists(label_path):
                self.image_paths.append(image_path)
                self.label_paths.append(label_path)
            else:
                print(f"Warning: No label found for {base_name}")
    
    def _process_mask(self, mask):
        """Fishyscapes: only pixels with value 1 are treated as positive (OOD objects)"""
        # Fishyscapes mask value meanings:
        # 0: background (normal)
        # 1: OOD object (anomaly) <- only this is treated as positive
        # 255: known object (normal)
        
        if len(mask.shape) == 3:
            # If RGB mask, use the first channel
            mask = mask[:, :, 0]
        
        # Only treat pixels with value 1 as positive
        binary_mask = (mask == 1).astype(bool)
        
        return binary_mask


class SegmentMeDataset(BaseAnomalyDataset):
    """Segment Me dataset (AnomalyTrack, ObstacleTrack)
    - Only validation series have labels
    - Label filename: validation*_labels_semantic_color.png (color version used)
    - Mask: only orange pixels are treated as OOD objects
    """
    
    def _load_data(self):
        original_dir = os.path.join(self.dataset_dir, 'original')
        label_dir = os.path.join(self.dataset_dir, 'labels')
        
        if not os.path.exists(original_dir) or not os.path.exists(label_dir):
            print(f"Warning: Directory not found - {original_dir} or {label_dir}")
            return
        
        # Find semantic labels for validation series (including color version)
        # AnomalyTrack: validation0000_labels_semantic_color.png
        # ObstacleTrack: validation_1_labels_semantic_color.png
        label_patterns = [
            'validation*_labels_semantic_color.png',  # color version
            'validation*_labels_semantic.png',        # standard version (fallback)
            'validation_*_labels_semantic_color.png', # ObstacleTrack color version
            'validation_*_labels_semantic.png'        # ObstacleTrack standard version (fallback)
        ]
        
        label_files = []
        for pattern in label_patterns:
            found_labels = glob(os.path.join(label_dir, pattern))
            label_files.extend(found_labels)
        
        # Remove duplicates (prefer color version for same base name)
        label_dict = {}
        for label_path in label_files:
            label_filename = os.path.basename(label_path)
            
            # Extract base name
            if '_labels_semantic_color' in label_filename:
                base_name = label_filename.replace('_labels_semantic_color.png', '')
                priority = 1  # color version takes priority
            elif '_labels_semantic' in label_filename:
                base_name = label_filename.replace('_labels_semantic.png', '')
                priority = 2  # standard version as fallback
            else:
                continue
            
            # Keep only the highest priority
            if base_name not in label_dict or priority < label_dict[base_name][1]:
                label_dict[base_name] = (label_path, priority)
        
        print(f"Found {len(label_dict)} unique label files")
        
        # Match images to labels
        for base_name, (label_path, _) in label_dict.items():
            # Find corresponding image file (try various extensions)
            possible_extensions = ['.jpg', '.jpeg', '.png', '.webp']
            image_path = None
            
            for ext in possible_extensions:
                potential_image = os.path.join(original_dir, f"{base_name}{ext}")
                if os.path.exists(potential_image):
                    image_path = potential_image
                    break
            
            if image_path:
                self.image_paths.append(image_path)
                self.label_paths.append(label_path)
            else:
                print(f"Warning: No image found for {base_name}")
    
    def _process_mask(self, mask):
        """Segment Me: only treat orange pixels as positive"""
        if len(mask.shape) == 3:
            # Detect orange pixels in RGB mask
            
            # Method 1: RGB-based orange detection (lenient range)
            # Orange range: R > 150, G > 30, G < 200, B < 100
            orange_mask_rgb = (mask[:, :, 0] > 150) & \
                              (mask[:, :, 1] > 30) & \
                              (mask[:, :, 1] < 200) & \
                              (mask[:, :, 2] < 100)
            
            # Method 2: HSV-based orange detection (more accurate)
            try:
                import cv2
                hsv = cv2.cvtColor(mask, cv2.COLOR_RGB2HSV)
                
                # Orange HSV range (OpenCV HSV: H=0-179, S=0-255, V=0-255)
                # Orange: H=10-25, S=100-255, V=100-255
                lower_orange = np.array([5, 50, 50])    # wider range
                upper_orange = np.array([30, 255, 255])
                
                hsv_orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
                orange_mask_hsv = (hsv_orange_mask > 0)
                
                # Use the method that finds more pixels
                if np.sum(orange_mask_hsv) > np.sum(orange_mask_rgb):
                    binary_mask = orange_mask_hsv.astype(bool)
                else:
                    binary_mask = orange_mask_rgb.astype(bool)
                    
            except ImportError:
                print("Warning: OpenCV not available, using RGB method only")
                binary_mask = orange_mask_rgb.astype(bool)
                
        else:
            # Grayscale mask case (unexpected)
            print("Warning: Grayscale mask detected in Segment Me dataset")
            binary_mask = (mask > 0).astype(bool)
        
        return binary_mask


class DatasetFactory:
    """Dataset factory class"""
    
    @staticmethod
    def create_dataset(dataset_dir, dataset_type=None):
        """Create appropriate dataset class based on dataset type"""
        
        if dataset_type is None:
            # Auto-detect type from path
            dataset_type = DatasetFactory._detect_dataset_type(dataset_dir)
        
        print(f"Creating dataset: {dataset_type} for {dataset_dir}")
        
        if dataset_type == 'road_anomaly':
            return RoadAnomalyDataset(dataset_dir)
        elif dataset_type == 'fishyscapes':
            return FishyscapesDataset(dataset_dir)
        elif dataset_type == 'segment_me':
            return SegmentMeDataset(dataset_dir)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    @staticmethod
    def _detect_dataset_type(dataset_dir):
        """Auto-detect dataset type from directory path"""
        dataset_dir_lower = dataset_dir.lower()
        
        if 'road_anomaly' in dataset_dir_lower:
            return 'road_anomaly'
        elif 'fishyscapes' in dataset_dir_lower:
            return 'fishyscapes'
        elif 'segment_me' in dataset_dir_lower or 'anomalytrack' in dataset_dir_lower or 'obstacletrack' in dataset_dir_lower:
            return 'segment_me'
        else:
            # Default to road_anomaly style
            print(f"Warning: Could not detect dataset type for {dataset_dir}, using road_anomaly as default")
            return 'road_anomaly'


def test_segment_me_datasets():
    """Test Segment Me datasets"""
    dataset_paths = [
        "./data/segment_me_val/dataset_AnomalyTrack",
        "./data/segment_me_val/dataset_ObstacleTrack"
    ]
    
    for dataset_dir in dataset_paths:
        if not os.path.exists(dataset_dir):
            print(f"Skipping {dataset_dir} (not found)")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing dataset: {dataset_dir}")
        print(f"{'='*60}")
        
        try:
            # Create dataset
            dataset = DatasetFactory.create_dataset(dataset_dir)
            
            if len(dataset) == 0:
                print("❌ No valid samples found!")
                continue
            
            print(f"✅ Dataset loaded successfully: {len(dataset)} samples")
            
            # Test first 3 samples
            for i in range(min(3, len(dataset))):
                print(f"\n--- Sample {i} ---")
                sample_info = dataset.get_sample_info(i)
                
                print(f"Name: {sample_info['image_name']}")
                print(f"Image size: {sample_info['image_size']}")
                print(f"Mask shape: {sample_info['mask_shape']}")
                print(f"Positive pixels: {sample_info['positive_pixels']}")
                print(f"Positive ratio: {sample_info['positive_ratio']:.4f}")
                
                # Test actual sample loading
                sample = dataset[i]
                
                # Check size consistency
                image_hw = sample['image'].size[::-1]  # PIL: (W, H) -> (H, W)
                mask_hw = sample['mask'].shape
                
                if image_hw == mask_hw:
                    print(f"✅ Image-mask size match: {image_hw}")
                else:
                    print(f"❌ Size mismatch: Image {image_hw} vs Mask {mask_hw}")
                
                # Check mask values
                mask_unique = np.unique(sample['mask'])
                print(f"Mask unique values: {mask_unique}")
                
                if len(mask_unique) == 2 and set(mask_unique) == {False, True}:
                    print("✅ Binary mask confirmed")
                elif len(mask_unique) == 1 and mask_unique[0] == False:
                    print("⚠️ No positive pixels found - check orange color detection")
                else:
                    print(f"⚠️ Unexpected mask values: {mask_unique}")
                    
        except Exception as e:
            print(f"❌ Error testing {dataset_dir}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_segment_me_datasets()