import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from ..mode_collapse_detector import ModeCollapseDetector, mode_collapse_detector

class BaseGAN:
    """Base class for all GAN implementations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.mode_collapse_detector = mode_collapse_detector
        self.training_metrics = []
        self.mode_collapse_history = []
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, Dict]:
        """Preprocess data for GAN training"""
        processed_data = data.copy()
        metadata = {'categorical_columns': [], 'numerical_columns': []}
        
        # Handle categorical columns
        for col in processed_data.columns:
            if processed_data[col].dtype == 'object' or processed_data[col].dtype == 'category':
                metadata['categorical_columns'].append(col)
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
                self.label_encoders[col] = le
            else:
                metadata['numerical_columns'].append(col)
        
        # Scale numerical columns
        if metadata['numerical_columns']:
            processed_data[metadata['numerical_columns']] = self.scaler.fit_transform(
                processed_data[metadata['numerical_columns']]
            )
        
        return torch.FloatTensor(processed_data.values), metadata
    
    def postprocess_data(self, synthetic_data: torch.Tensor, metadata: Dict) -> pd.DataFrame:
        """Convert synthetic data back to original format"""
        synthetic_df = pd.DataFrame(synthetic_data.numpy(), columns=metadata.get('original_columns', []))
        
        # Inverse scale numerical columns
        if metadata['numerical_columns']:
            synthetic_df[metadata['numerical_columns']] = self.scaler.inverse_transform(
                synthetic_df[metadata['numerical_columns']]
            )
        
        # Inverse transform categorical columns
        for col in metadata['categorical_columns']:
            if col in self.label_encoders:
                synthetic_df[col] = self.label_encoders[col].inverse_transform(
                    synthetic_df[col].astype(int)
                )
        
        return synthetic_df
    
    def generate(self, data: pd.DataFrame, num_samples: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic data - to be implemented by subclasses"""
        raise NotImplementedError
    
    def generate_with_monitoring(self, data: pd.DataFrame, num_samples: Optional[int] = None, 
                                original_data: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generate synthetic data with mode collapse monitoring
        
        Args:
            data: Training data
            num_samples: Number of samples to generate
            original_data: Original data for comparison (if different from training data)
            
        Returns:
            Tuple of (synthetic_data, monitoring_results)
        """
        # Generate synthetic data
        synthetic_data = self.generate(data, num_samples)
        
        # Monitor for mode collapse
        comparison_data = original_data if original_data is not None else data
        mode_collapse_metrics = self.mode_collapse_detector.detect_mode_collapse(
            comparison_data, synthetic_data
        )
        
        # Store monitoring results
        self.mode_collapse_history.append(mode_collapse_metrics)
        
        monitoring_results = {
            'mode_collapse_detected': mode_collapse_metrics.mode_collapse_detected,
            'severity': mode_collapse_metrics.severity,
            'diversity_score': mode_collapse_metrics.diversity_score,
            'uniqueness_ratio': mode_collapse_metrics.uniqueness_ratio,
            'recommendations': mode_collapse_metrics.recommendations,
            'should_retrain': self._should_retrain_based_on_collapse()
        }
        
        return synthetic_data, monitoring_results
    
    def _should_retrain_based_on_collapse(self) -> bool:
        """Determine if retraining should be triggered based on mode collapse history"""
        if len(self.mode_collapse_history) < 3:
            return False
        
        # Check recent mode collapse history
        recent_metrics = self.mode_collapse_history[-3:]
        return self.mode_collapse_detector.should_trigger_retraining(recent_metrics, threshold_epochs=3)
    
    def get_prevention_strategies(self) -> Dict[str, Any]:
        """Get prevention strategies based on current mode collapse status"""
        if not self.mode_collapse_history:
            return {}
        
        latest_metrics = self.mode_collapse_history[-1]
        return self.mode_collapse_detector.get_prevention_strategies(latest_metrics.severity)
    
    def monitor_training_step(self, epoch: int, generator_loss: float, discriminator_loss: float,
                            synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """Monitor a single training step for mode collapse indicators"""
        training_metrics = self.mode_collapse_detector.monitor_training(
            epoch, generator_loss, discriminator_loss, synthetic_data
        )
        
        self.training_metrics.append(training_metrics)
        
        return {
            'epoch': epoch,
            'diversity_score': training_metrics.diversity_score,
            'mode_collapse_risk': training_metrics.mode_collapse_risk,
            'unique_samples': training_metrics.unique_samples,
            'synthetic_samples': training_metrics.synthetic_samples
        }
    
    def get_mode_collapse_summary(self) -> Dict[str, Any]:
        """Get summary of mode collapse monitoring"""
        return self.mode_collapse_detector.get_summary_report()
    
    def generate_mode_collapse_visualization(self, synthetic_data: pd.DataFrame, 
                                           save_path: Optional[str] = None) -> str:
        """Generate visualization for mode collapse analysis"""
        if save_path is None:
            save_path = f'mode_collapse_analysis_{self.__class__.__name__}.png'
        
        return self.mode_collapse_detector.generate_visualization(synthetic_data, save_path) 