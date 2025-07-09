import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Optional
from .base_gan import BaseGAN

class SMOTECTGAN(BaseGAN):
    """SMOTE-enhanced CTGAN for imbalanced datasets"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        config = config or {}
        self.k_neighbors = config.get('k_neighbors', 5)
        self.smote_ratio = config.get('smote_ratio', 0.5)
        
    def _smote_oversample(self, data: np.ndarray, target_col: str = None) -> np.ndarray:
        """Apply SMOTE oversampling"""
        if target_col is None:
            # Find the column with the most unique values (likely the target)
            unique_counts = [len(np.unique(data[:, i])) for i in range(data.shape[1])]
            target_col = np.argmin(unique_counts)
        
        # Get unique classes and their counts
        classes, counts = np.unique(data[:, target_col], return_counts=True)
        majority_class = classes[np.argmax(counts)]
        minority_classes = classes[counts < np.max(counts)]
        
        synthetic_samples = []
        
        for minority_class in minority_classes:
            minority_samples = data[data[:, target_col] == minority_class]
            minority_count = len(minority_samples)
            majority_count = np.max(counts)
            
            # Calculate how many synthetic samples to generate
            samples_needed = majority_count - minority_count
            
            for _ in range(samples_needed):
                # Randomly select a minority sample
                sample = minority_samples[np.random.randint(minority_count)]
                
                # Find k nearest neighbors
                distances = np.linalg.norm(minority_samples - sample, axis=1)
                nearest_indices = np.argsort(distances)[1:self.k_neighbors + 1]
                
                # Randomly select a neighbor
                neighbor = minority_samples[np.random.choice(nearest_indices)]
                
                # Generate synthetic sample
                synthetic_sample = sample + np.random.random() * (neighbor - sample)
                synthetic_samples.append(synthetic_sample)
        
        return np.vstack([data, synthetic_samples])
    
    def generate(self, data: pd.DataFrame, num_samples: int = None) -> pd.DataFrame:
        if num_samples is None:
            num_samples = len(data)
        
        # Preprocess data
        processed_data, metadata = self.preprocess_data(data)
        metadata['original_columns'] = data.columns.tolist()
        
        # Apply SMOTE
        balanced_data = self._smote_oversample(processed_data.numpy())
        
        # Generate additional samples if needed
        if len(balanced_data) < num_samples:
            additional_needed = num_samples - len(balanced_data)
            additional_samples = balanced_data[np.random.choice(len(balanced_data), additional_needed)]
            balanced_data = np.vstack([balanced_data, additional_samples])
        
        # Take only the required number of samples
        balanced_data = balanced_data[:num_samples]
        
        return self.postprocess_data(torch.FloatTensor(balanced_data), metadata) 