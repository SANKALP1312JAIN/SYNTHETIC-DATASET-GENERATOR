import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from .base_gan import BaseGAN

class DefaultGAN(BaseGAN):
    """Default GAN for unknown data types"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
    def generate(self, data: Any, num_samples: Optional[int] = None) -> Any:
        """Generate synthetic data using a simple approach"""
        if num_samples is None:
            num_samples = len(data) if hasattr(data, '__len__') else 100
        
        if isinstance(data, pd.DataFrame):
            # For tabular data, use simple resampling with noise
            synthetic_data = []
            for _ in range(num_samples):
                row = data.iloc[np.random.randint(len(data))].copy()
                # Add small noise to numerical columns
                for col in data.select_dtypes(include=[np.number]).columns:
                    row[col] += np.random.normal(0, 0.01)
                synthetic_data.append(row)
            return pd.DataFrame(synthetic_data)
        
        elif isinstance(data, list):
            # For list data, use resampling
            return [data[np.random.randint(len(data))] for _ in range(num_samples)]
        
        else:
            # For other types, return the original data
            return data 