import pandas as pd
import numpy as np
import torch
from typing import Dict, Any, Optional
from .base_gan import BaseGAN

class DoppelGANger(BaseGAN):
    """DoppelGANger for time series data"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        config = config or {}
        self.sequence_length = config.get('sequence_length', 10)
        self.hidden_dim = config.get('hidden_dim', 128)
        
    def _prepare_sequences(self, data: pd.Series) -> np.ndarray:
        """Convert time series to sequences"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data.iloc[i:i + self.sequence_length].values)
        return np.array(sequences)
    
    def generate(self, data: pd.DataFrame, num_samples: Optional[int] = None) -> pd.DataFrame:
        if num_samples is None:
            num_samples = len(data)
        
        # Assume first column is the time series
        time_series = data.iloc[:, 0]
        
        # Prepare sequences
        sequences = self._prepare_sequences(time_series)
        
        # Generate synthetic sequences
        synthetic_sequences = []
        for _ in range(num_samples):
            # Simple approach: sample and add noise
            base_sequence = sequences[np.random.randint(len(sequences))]
            noise = np.random.normal(0, 0.1, base_sequence.shape)
            synthetic_sequence = base_sequence + noise
            synthetic_sequences.append(synthetic_sequence)
        
        # Convert back to DataFrame
        synthetic_data = []
        for seq in synthetic_sequences:
            synthetic_data.extend(seq)
        
        # Create DataFrame with same structure - fix the hashable key issue
        first_col_name = str(data.columns[0])
        result_df = pd.DataFrame({
            first_col_name: synthetic_data[:num_samples]
        })
        
        # Add other columns if they exist
        for col in data.columns[1:]:
            result_df[str(col)] = data[col].iloc[:num_samples].values
        
        return result_df 