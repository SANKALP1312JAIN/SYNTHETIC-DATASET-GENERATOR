"""
Mode Collapse Detection and Prevention System

This module provides comprehensive mode collapse monitoring including:
- Diversity metrics calculation
- Mode collapse detection algorithms
- Preventive measures and retraining triggers
- Quality monitoring and feedback loops
- Adaptive training strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import Counter
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModeCollapseMetrics:
    """Metrics for mode collapse detection"""
    timestamp: datetime
    diversity_score: float
    uniqueness_ratio: float
    cluster_score: float
    distribution_similarity: float
    mode_collapse_detected: bool
    severity: str  # 'none', 'mild', 'moderate', 'severe'
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingMetrics:
    """Training metrics for monitoring"""
    epoch: int
    generator_loss: float
    discriminator_loss: float
    diversity_score: float
    synthetic_samples: int
    unique_samples: int
    mode_collapse_risk: float

class ModeCollapseDetector:
    """Mode collapse detection and prevention system"""
    
    def __init__(self):
        self.mode_collapse_thresholds = {
            'diversity_score': 0.7,  # Minimum acceptable diversity
            'uniqueness_ratio': 0.8,  # Minimum unique samples ratio
            'cluster_score': 0.6,     # Minimum clustering quality
            'distribution_similarity': 0.85  # Maximum similarity to detect collapse
        }
        
        self.training_history: List[TrainingMetrics] = []
        self.mode_collapse_history: List[ModeCollapseMetrics] = []
        self.prevention_strategies = {
            'gradient_penalty': True,
            'label_smoothing': True,
            'noise_injection': True,
            'adaptive_learning_rate': True,
            'batch_diversity_monitoring': True
        }
    
    def detect_mode_collapse(self, original_data: pd.DataFrame, 
                           synthetic_data: pd.DataFrame,
                           training_metrics: Optional[Dict[str, Any]] = None) -> ModeCollapseMetrics:
        """
        Detect mode collapse in synthetic data
        
        Args:
            original_data: Original training data
            synthetic_data: Generated synthetic data
            training_metrics: Optional training metrics
            
        Returns:
            ModeCollapseMetrics object with detection results
        """
        logger.info(f"Detecting mode collapse for {len(synthetic_data)} synthetic samples")
        
        # Calculate various diversity metrics
        diversity_score = self._calculate_diversity_score(synthetic_data)
        uniqueness_ratio = self._calculate_uniqueness_ratio(synthetic_data)
        cluster_score = self._calculate_cluster_score(synthetic_data)
        distribution_similarity = self._calculate_distribution_similarity(original_data, synthetic_data)
        
        # Determine if mode collapse is detected
        mode_collapse_detected = self._evaluate_mode_collapse(
            diversity_score, uniqueness_ratio, cluster_score, distribution_similarity
        )
        
        # Determine severity
        severity = self._determine_severity(
            diversity_score, uniqueness_ratio, cluster_score, distribution_similarity
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            mode_collapse_detected, severity, training_metrics
        )
        
        metrics = ModeCollapseMetrics(
            timestamp=datetime.now(),
            diversity_score=diversity_score,
            uniqueness_ratio=uniqueness_ratio,
            cluster_score=cluster_score,
            distribution_similarity=distribution_similarity,
            mode_collapse_detected=mode_collapse_detected,
            severity=severity,
            recommendations=recommendations,
            metadata={
                'original_samples': len(original_data),
                'synthetic_samples': len(synthetic_data),
                'training_metrics': training_metrics
            }
        )
        
        self.mode_collapse_history.append(metrics)
        
        if mode_collapse_detected:
            logger.warning(f"Mode collapse detected! Severity: {severity}")
        else:
            logger.info("No mode collapse detected")
        
        return metrics
    
    def _calculate_diversity_score(self, synthetic_data: pd.DataFrame) -> float:
        """Calculate diversity score based on sample variety"""
        if len(synthetic_data) == 0:
            return 0.0
        
        # Calculate diversity using multiple approaches
        
        # 1. Numerical diversity (for numerical columns)
        numerical_cols = synthetic_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            numerical_data = synthetic_data[numerical_cols]
            if isinstance(numerical_data, pd.Series):
                numerical_data = numerical_data.to_frame()
            numerical_diversity = self._calculate_numerical_diversity(numerical_data)
        else:
            numerical_diversity = 1.0
        
        # 2. Categorical diversity (for categorical columns)
        categorical_cols = synthetic_data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            categorical_data = synthetic_data[categorical_cols]
            if isinstance(categorical_data, pd.Series):
                categorical_data = categorical_data.to_frame()
            categorical_diversity = self._calculate_categorical_diversity(categorical_data)
        else:
            categorical_diversity = 1.0
        
        # 3. Overall sample diversity
        sample_diversity = self._calculate_sample_diversity(synthetic_data)
        
        # Combine diversities (weighted average)
        diversity_score = (numerical_diversity * 0.4 + 
                          categorical_diversity * 0.4 + 
                          sample_diversity * 0.2)
        
        return min(diversity_score, 1.0)
    
    def _calculate_numerical_diversity(self, numerical_data: pd.DataFrame) -> float:
        """Calculate diversity for numerical columns"""
        if len(numerical_data) == 0:
            return 0.0
        
        # Calculate coefficient of variation for each column
        cvs = []
        for col in numerical_data.columns:
            std = numerical_data[col].std()
            mean = numerical_data[col].mean()
            if mean != 0:
                cv = std / abs(mean)
                cvs.append(cv)
        
        if not cvs:
            return 0.0
        
        # Normalize CVs to [0, 1] range (assuming CV > 0.1 is good diversity)
        normalized_cvs = [min(cv / 0.1, 1.0) for cv in cvs]
        return float(np.mean(normalized_cvs))
    
    def _calculate_categorical_diversity(self, categorical_data: pd.DataFrame) -> float:
        """Calculate diversity for categorical columns"""
        if len(categorical_data) == 0:
            return 0.0
        
        diversities = []
        for col in categorical_data.columns:
            value_counts = categorical_data[col].value_counts()
            if len(value_counts) > 1:
                # Calculate entropy-based diversity
                total = len(categorical_data)
                entropy = -sum((count/total) * np.log2(count/total) for count in value_counts)
                max_entropy = np.log2(len(value_counts))
                if max_entropy > 0:
                    diversity = entropy / max_entropy
                    diversities.append(diversity)
                else:
                    diversities.append(0.0)
            else:
                diversities.append(0.0)
        
        return float(np.mean(diversities)) if diversities else 0.0
    
    def _calculate_sample_diversity(self, data: pd.DataFrame) -> float:
        """Calculate overall sample diversity"""
        if len(data) < 2:
            return 0.0
        
        # Use clustering to assess sample diversity
        try:
            # Convert to numerical representation for clustering
            numerical_data = data.select_dtypes(include=[np.number])
            if len(numerical_data.columns) == 0:
                # If no numerical columns, use one-hot encoding
                numerical_data = pd.get_dummies(data, drop_first=True)
            if len(numerical_data.columns) == 0:
                return 0.0
            # Normalize data
            normalized_data = (numerical_data - numerical_data.mean()) / numerical_data.std()
            normalized_data = normalized_data.fillna(0)
            # Use silhouette score to measure clustering quality
            if len(data) > 10:
                n_clusters = min(5, len(data) // 10)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="10")
                cluster_labels = kmeans.fit_predict(normalized_data.to_numpy())
                if len(np.unique(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(normalized_data.to_numpy(), cluster_labels)
                    return max(0, float(silhouette_avg))  # Ensure non-negative
                else:
                    return 0.0
            else:
                return 0.5  # Default for small datasets
                
        except Exception as e:
            logger.warning(f"Error calculating sample diversity: {e}")
            return 0.5
    
    def _calculate_uniqueness_ratio(self, synthetic_data: pd.DataFrame) -> float:
        """Calculate ratio of unique samples"""
        if len(synthetic_data) == 0:
            return 0.0
        
        # Count unique samples
        unique_samples = synthetic_data.drop_duplicates()
        uniqueness_ratio = len(unique_samples) / len(synthetic_data)
        
        return uniqueness_ratio
    
    def _calculate_cluster_score(self, synthetic_data: pd.DataFrame) -> float:
        """Calculate clustering quality score"""
        if len(synthetic_data) < 10:
            return 0.5
        
        try:
            # Convert to numerical representation
            numerical_data = synthetic_data.select_dtypes(include=[np.number])
            if len(numerical_data.columns) == 0:
                numerical_data = pd.get_dummies(synthetic_data, drop_first=True)
            if len(numerical_data.columns) == 0:
                return 0.5
            # Normalize data
            normalized_data = (numerical_data - numerical_data.mean()) / numerical_data.std()
            normalized_data = normalized_data.fillna(0)
            # Calculate Calinski-Harabasz score
            n_clusters = min(5, len(synthetic_data) // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="10")
            cluster_labels = kmeans.fit_predict(normalized_data.to_numpy())
            if len(np.unique(cluster_labels)) > 1:
                ch_score = calinski_harabasz_score(normalized_data.to_numpy(), cluster_labels)
                # Normalize CH score (typically ranges from 0 to 1000+)
                normalized_score = min(ch_score / 100, 1.0)
                return float(normalized_score)
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error calculating cluster score: {e}")
            return 0.5
    
    def _calculate_distribution_similarity(self, original_data: pd.DataFrame, 
                                         synthetic_data: pd.DataFrame) -> float:
        """Calculate similarity between original and synthetic distributions"""
        if len(original_data) == 0 or len(synthetic_data) == 0:
            return 0.0
        
        similarities = []
        
        # Compare distributions for each column
        for col in original_data.columns:
            if col in synthetic_data.columns:
                orig_col = original_data[col]
                synth_col = synthetic_data[col]
                
                if orig_col.dtype in ['object', 'category']:
                    # Categorical similarity
                    if isinstance(orig_col, pd.DataFrame):
                        orig_col = orig_col.iloc[:, 0]
                    if isinstance(synth_col, pd.DataFrame):
                        synth_col = synth_col.iloc[:, 0]
                    similarity = self._calculate_categorical_similarity(orig_col, synth_col)
                else:
                    # Numerical similarity
                    if isinstance(orig_col, pd.DataFrame):
                        orig_col = orig_col.iloc[:, 0]
                    if isinstance(synth_col, pd.DataFrame):
                        synth_col = synth_col.iloc[:, 0]
                    similarity = self._calculate_numerical_similarity(orig_col, synth_col)
                
                similarities.append(similarity)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _calculate_categorical_similarity(self, orig_col: pd.Series, synth_col: pd.Series) -> float:
        """Calculate similarity for categorical columns"""
        orig_counts = orig_col.value_counts(normalize=True)
        synth_counts = synth_col.value_counts(normalize=True)
        
        # Get all unique values
        all_values = set(orig_counts.index) | set(synth_counts.index)
        
        # Calculate cosine similarity
        orig_vec = np.array([orig_counts.get(val, 0) for val in all_values])
        synth_vec = np.array([synth_counts.get(val, 0) for val in all_values])
        
        # Normalize vectors
        orig_norm = np.linalg.norm(orig_vec)
        synth_norm = np.linalg.norm(synth_vec)
        
        if orig_norm == 0 or synth_norm == 0:
            return 0.0
        
        similarity = np.dot(orig_vec, synth_vec) / (orig_norm * synth_norm)
        return similarity
    
    def _calculate_numerical_similarity(self, orig_col: pd.Series, synth_col: pd.Series) -> float:
        """Calculate similarity for numerical columns"""
        # Compare basic statistics
        orig_stats = {
            'mean': orig_col.mean(),
            'std': orig_col.std(),
            'skew': orig_col.skew(),
            'kurt': orig_col.kurtosis()
        }
        
        synth_stats = {
            'mean': synth_col.mean(),
            'std': synth_col.std(),
            'skew': synth_col.skew(),
            'kurt': synth_col.kurtosis()
        }
        
        # Calculate similarity for each statistic
        similarities = []
        for stat in ['mean', 'std', 'skew', 'kurt']:
            orig_val = orig_stats[stat]
            synth_val = synth_stats[stat]
            
            if bool(pd.isna(orig_val)) or bool(pd.isna(synth_val)):
                similarities.append(0.0)
            else:
                # Normalize difference
                if orig_val != 0:
                    diff = abs(orig_val - synth_val) / abs(orig_val)
                    similarity = max(0, 1 - diff)
                else:
                    similarity = 1.0 if synth_val == 0 else 0.0
                
                similarities.append(similarity)
        
        return float(np.mean(similarities))
    
    def _evaluate_mode_collapse(self, diversity_score: float, uniqueness_ratio: float,
                               cluster_score: float, distribution_similarity: float) -> bool:
        """Evaluate if mode collapse is detected"""
        thresholds = self.mode_collapse_thresholds
        
        # Check if any metric indicates mode collapse
        diversity_collapse = diversity_score < thresholds['diversity_score']
        uniqueness_collapse = uniqueness_ratio < thresholds['uniqueness_ratio']
        cluster_collapse = cluster_score < thresholds['cluster_score']
        similarity_collapse = distribution_similarity > thresholds['distribution_similarity']
        
        # Mode collapse is detected if multiple indicators are present
        collapse_indicators = [diversity_collapse, uniqueness_collapse, 
                             cluster_collapse, similarity_collapse]
        
        return sum(collapse_indicators) >= 2
    
    def _determine_severity(self, diversity_score: float, uniqueness_ratio: float,
                           cluster_score: float, distribution_similarity: float) -> str:
        """Determine severity of mode collapse"""
        # Calculate overall risk score
        risk_score = (
            (1 - diversity_score) * 0.3 +
            (1 - uniqueness_ratio) * 0.3 +
            (1 - cluster_score) * 0.2 +
            distribution_similarity * 0.2
        )
        
        if risk_score < 0.2:
            return 'none'
        elif risk_score < 0.4:
            return 'mild'
        elif risk_score < 0.6:
            return 'moderate'
        else:
            return 'severe'
    
    def _generate_recommendations(self, mode_collapse_detected: bool, severity: str,
                                training_metrics: Optional[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for preventing mode collapse"""
        recommendations = []
        
        if mode_collapse_detected:
            recommendations.append("Mode collapse detected - immediate action required")
            
            if severity in ['moderate', 'severe']:
                recommendations.append("Consider retraining with different hyperparameters")
                recommendations.append("Increase generator learning rate")
                recommendations.append("Add gradient penalty to discriminator")
                recommendations.append("Implement label smoothing")
                recommendations.append("Add noise injection during training")
            
            if severity == 'severe':
                recommendations.append("Switch to a different GAN architecture")
                recommendations.append("Increase training epochs significantly")
                recommendations.append("Use data augmentation techniques")
        
        # General recommendations
        recommendations.append("Monitor diversity metrics during training")
        recommendations.append("Use adaptive learning rates")
        recommendations.append("Implement early stopping based on diversity")
        
        return recommendations
    
    def monitor_training(self, epoch: int, generator_loss: float, discriminator_loss: float,
                        synthetic_data: pd.DataFrame) -> TrainingMetrics:
        """Monitor training progress for mode collapse indicators"""
        diversity_score = self._calculate_diversity_score(synthetic_data)
        uniqueness_ratio = self._calculate_uniqueness_ratio(synthetic_data)
        
        # Calculate mode collapse risk
        mode_collapse_risk = 1 - (diversity_score * 0.7 + uniqueness_ratio * 0.3)
        
        metrics = TrainingMetrics(
            epoch=epoch,
            generator_loss=generator_loss,
            discriminator_loss=discriminator_loss,
            diversity_score=diversity_score,
            synthetic_samples=len(synthetic_data),
            unique_samples=len(synthetic_data.drop_duplicates()),
            mode_collapse_risk=mode_collapse_risk
        )
        
        self.training_history.append(metrics)
        
        # Log warning if risk is high
        if mode_collapse_risk > 0.7:
            logger.warning(f"High mode collapse risk detected at epoch {epoch}: {mode_collapse_risk:.3f}")
        
        return metrics
    
    def get_prevention_strategies(self, severity: str) -> Dict[str, Any]:
        """Get prevention strategies based on severity"""
        strategies = {}
        
        if severity in ['mild', 'moderate', 'severe']:
            strategies['gradient_penalty'] = {
                'enabled': True,
                'lambda_gp': 10.0,
                'description': 'Add gradient penalty to stabilize training'
            }
            
            strategies['label_smoothing'] = {
                'enabled': True,
                'smoothing_factor': 0.1,
                'description': 'Use label smoothing to prevent overconfidence'
            }
            
            strategies['noise_injection'] = {
                'enabled': True,
                'noise_std': 0.01,
                'description': 'Add noise to prevent mode collapse'
            }
            
            if severity in ['moderate', 'severe']:
                strategies['adaptive_learning_rate'] = {
                    'enabled': True,
                    'lr_factor': 0.5,
                    'description': 'Reduce learning rate to stabilize training'
                }
                
                strategies['batch_diversity_monitoring'] = {
                    'enabled': True,
                    'min_diversity': 0.6,
                    'description': 'Monitor batch diversity during training'
                }
        
        return strategies
    
    def should_trigger_retraining(self, recent_metrics: List[ModeCollapseMetrics], 
                                threshold_epochs: int = 5) -> bool:
        """Determine if retraining should be triggered"""
        if len(recent_metrics) < threshold_epochs:
            return False
        
        # Check if mode collapse has been detected consistently
        recent_collapses = [m for m in recent_metrics[-threshold_epochs:] 
                           if m.mode_collapse_detected]
        
        if len(recent_collapses) >= threshold_epochs * 0.6:  # 60% of recent epochs
            return True
        
        # Check if severity is consistently high
        high_severity = [m for m in recent_metrics[-threshold_epochs:] 
                        if m.severity in ['moderate', 'severe']]
        
        if len(high_severity) >= threshold_epochs * 0.5:  # 50% of recent epochs
            return True
        
        return False
    
    def generate_visualization(self, synthetic_data: pd.DataFrame, 
                             save_path: str = 'mode_collapse_analysis.png') -> str:
        """Generate visualization for mode collapse analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Diversity score over time
            if self.training_history:
                epochs = [m.epoch for m in self.training_history]
                diversity_scores = [m.diversity_score for m in self.training_history]
                axes[0, 0].plot(epochs, diversity_scores, 'b-', linewidth=2)
                axes[0, 0].axhline(y=self.mode_collapse_thresholds['diversity_score'], 
                                 color='r', linestyle='--', label='Threshold')
                axes[0, 0].set_title('Diversity Score Over Time')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Diversity Score')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Mode collapse risk over time
            if self.training_history:
                risk_scores = [m.mode_collapse_risk for m in self.training_history]
                axes[0, 1].plot(epochs, risk_scores, 'r-', linewidth=2)
                axes[0, 1].axhline(y=0.7, color='orange', linestyle='--', label='High Risk Threshold')
                axes[0, 1].set_title('Mode Collapse Risk Over Time')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Risk Score')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Sample distribution (if numerical data available)
            numerical_cols = synthetic_data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) >= 2:
                axes[1, 0].scatter(synthetic_data[numerical_cols[0]], 
                                 synthetic_data[numerical_cols[1]], 
                                 alpha=0.6, s=20)
                axes[1, 0].set_title('Synthetic Data Distribution')
                axes[1, 0].set_xlabel(numerical_cols[0])
                axes[1, 0].set_ylabel(numerical_cols[1])
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Mode collapse history
            if self.mode_collapse_history:
                timestamps = [m.timestamp for m in self.mode_collapse_history]
                severities = [m.severity for m in self.mode_collapse_history]
                severity_map = {'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3}
                severity_values = [severity_map[s] for s in severities]
                
                axes[1, 1].plot(timestamps, severity_values, 'g-o', linewidth=2, markersize=6)
                axes[1, 1].set_title('Mode Collapse Severity Over Time')
                axes[1, 1].set_xlabel('Time')
                axes[1, 1].set_ylabel('Severity Level')
                axes[1, 1].set_yticks([0, 1, 2, 3])
                axes[1, 1].set_yticklabels(['None', 'Mild', 'Moderate', 'Severe'])
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return save_path
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return ""
    
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report of mode collapse monitoring"""
        if not self.mode_collapse_history:
            return {"message": "No mode collapse monitoring data available"}
        
        total_checks = len(self.mode_collapse_history)
        collapse_detected = sum(1 for m in self.mode_collapse_history if m.mode_collapse_detected)
        collapse_rate = collapse_detected / total_checks if total_checks > 0 else 0
        
        severity_counts = Counter(m.severity for m in self.mode_collapse_history)
        
        avg_diversity = np.mean([m.diversity_score for m in self.mode_collapse_history])
        avg_uniqueness = np.mean([m.uniqueness_ratio for m in self.mode_collapse_history])
        
        return {
            "total_checks": total_checks,
            "collapse_detected": collapse_detected,
            "collapse_rate": collapse_rate,
            "severity_distribution": dict(severity_counts),
            "average_diversity_score": avg_diversity,
            "average_uniqueness_ratio": avg_uniqueness,
            "latest_metrics": self.mode_collapse_history[-1] if self.mode_collapse_history else None,
            "recommendations": self.mode_collapse_history[-1].recommendations if self.mode_collapse_history else []
        }

# Global mode collapse detector instance
mode_collapse_detector = ModeCollapseDetector()
