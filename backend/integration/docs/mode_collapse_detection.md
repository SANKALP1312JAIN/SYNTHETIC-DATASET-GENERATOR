# Mode Collapse Detection and Prevention System

## Overview

The Mode Collapse Detection and Prevention System is a comprehensive monitoring solution designed to detect, analyze, and prevent mode collapse in GAN-based synthetic data generation. Mode collapse occurs when a GAN generates a limited variety of samples, failing to capture the full diversity of the original data distribution.

## Features

### 1. Multi-Metric Detection
The system uses multiple complementary metrics to detect mode collapse:

- **Diversity Score**: Measures the variety of generated samples across numerical and categorical features
- **Uniqueness Ratio**: Calculates the proportion of unique samples in the generated data
- **Cluster Score**: Evaluates clustering quality using Calinski-Harabasz score
- **Distribution Similarity**: Compares statistical properties between original and synthetic data

### 2. Severity Assessment
Mode collapse is classified into four severity levels:

- **None**: No mode collapse detected
- **Mild**: Minor diversity issues, basic prevention strategies recommended
- **Moderate**: Significant diversity problems, advanced prevention needed
- **Severe**: Critical mode collapse, immediate intervention required

### 3. Real-time Training Monitoring
Continuous monitoring during GAN training:

- Tracks diversity metrics across training epochs
- Monitors mode collapse risk progression
- Provides early warning signals
- Enables adaptive training strategies

### 4. Prevention Strategies
Automated prevention recommendations based on severity:

#### Mild Mode Collapse:
- Gradient penalty implementation
- Label smoothing
- Noise injection

#### Moderate Mode Collapse:
- Adaptive learning rate adjustment
- Batch diversity monitoring
- Enhanced regularization

#### Severe Mode Collapse:
- Architecture switching recommendations
- Extended training epochs
- Data augmentation techniques

## Usage

### Basic Mode Collapse Detection

```python
from src.agent.mode_collapse_detector import mode_collapse_detector

# Detect mode collapse
metrics = mode_collapse_detector.detect_mode_collapse(
    original_data, synthetic_data
)

print(f"Mode Collapse Detected: {metrics.mode_collapse_detected}")
print(f"Severity: {metrics.severity}")
print(f"Diversity Score: {metrics.diversity_score:.3f}")
```

### Training Progress Monitoring

```python
# Monitor during training
for epoch in range(num_epochs):
    # Generate synthetic data for this epoch
    synthetic_data = gan.generate(training_data)
    
    # Monitor training progress
    training_metrics = mode_collapse_detector.monitor_training(
        epoch=epoch,
        generator_loss=gen_loss,
        discriminator_loss=disc_loss,
        synthetic_data=synthetic_data
    )
    
    # Check if retraining is needed
    if training_metrics.mode_collapse_risk > 0.7:
        print(f"High mode collapse risk at epoch {epoch}")
```

### Integration with SyntheticDataAgent

```python
from src.agent.synthetic_data_agent import SyntheticDataAgent

agent = SyntheticDataAgent()

# Generate synthetic data with monitoring
synthetic_data, generation_metadata = agent.generate_synthetic_data(
    data, 'DP-CTGAN', 1000, 'healthcare', 'high'
)

# Detect mode collapse
collapse_metrics = agent.detect_mode_collapse(original_data, synthetic_data)

# Get prevention strategies
strategies = agent.get_mode_collapse_prevention_strategies(
    collapse_metrics.severity
)

# Generate visualization
viz_path = agent.generate_mode_collapse_visualization(synthetic_data)
```

## Metrics Explained

### Diversity Score
The diversity score combines three components:

1. **Numerical Diversity**: Coefficient of variation for numerical features
2. **Categorical Diversity**: Entropy-based diversity for categorical features  
3. **Sample Diversity**: Clustering-based diversity using silhouette score

**Formula**: `Diversity = 0.4 × Numerical + 0.4 × Categorical + 0.2 × Sample`

### Uniqueness Ratio
Measures the proportion of unique samples in the generated data:

**Formula**: `Uniqueness = Unique_Samples / Total_Samples`

### Cluster Score
Uses Calinski-Harabasz score to evaluate clustering quality:

**Formula**: `CH_Score = (Between_Cluster_Variance / Within_Cluster_Variance) × (n - k) / (k - 1)`

### Distribution Similarity
Compares statistical properties between original and synthetic data:

- **Numerical**: Mean, standard deviation, skewness, kurtosis
- **Categorical**: Value distribution using cosine similarity

## Prevention Strategies

### Gradient Penalty
Adds regularization to prevent mode collapse:

```python
# Implementation example
lambda_gp = 10.0
gradient_penalty = lambda_gp * torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
```

### Label Smoothing
Prevents overconfidence in discriminator:

```python
# Implementation example
smoothing_factor = 0.1
real_labels = torch.ones(batch_size) * (1 - smoothing_factor)
fake_labels = torch.zeros(batch_size) + smoothing_factor
```

### Noise Injection
Adds controlled noise during training:

```python
# Implementation example
noise_std = 0.01
noise = torch.randn_like(input_data) * noise_std
noisy_data = input_data + noise
```

### Adaptive Learning Rate
Dynamically adjusts learning rates based on diversity:

```python
# Implementation example
if diversity_score < threshold:
    learning_rate *= 0.5  # Reduce learning rate
```

## Visualization

The system generates comprehensive visualizations including:

1. **Diversity Score Over Time**: Tracks diversity progression across epochs
2. **Mode Collapse Risk**: Shows risk level changes during training
3. **Sample Distribution**: Scatter plots of synthetic data distribution
4. **Severity Timeline**: Historical mode collapse severity tracking

## Configuration

### Thresholds
Default thresholds can be customized:

```python
mode_collapse_detector.mode_collapse_thresholds = {
    'diversity_score': 0.7,      # Minimum acceptable diversity
    'uniqueness_ratio': 0.8,     # Minimum unique samples ratio
    'cluster_score': 0.6,        # Minimum clustering quality
    'distribution_similarity': 0.85  # Maximum similarity to detect collapse
}
```

### Prevention Strategy Configuration

```python
mode_collapse_detector.prevention_strategies = {
    'gradient_penalty': True,
    'label_smoothing': True,
    'noise_injection': True,
    'adaptive_learning_rate': True,
    'batch_diversity_monitoring': True
}
```

## Best Practices

### 1. Regular Monitoring
- Monitor diversity metrics every 10-20 epochs
- Set up automated alerts for high-risk scenarios
- Track long-term trends in mode collapse patterns

### 2. Early Intervention
- Implement prevention strategies at first signs of mode collapse
- Use adaptive training parameters based on diversity scores
- Consider architecture changes for persistent issues

### 3. Data Quality
- Ensure original data has sufficient diversity
- Preprocess data to remove extreme outliers
- Balance categorical variables if necessary

### 4. Training Stability
- Use gradient clipping to prevent exploding gradients
- Implement proper weight initialization
- Monitor loss convergence patterns

## Troubleshooting

### Common Issues

1. **False Positives**: Adjust thresholds based on domain-specific requirements
2. **High Computational Cost**: Reduce monitoring frequency for large datasets
3. **Memory Issues**: Process data in batches for large-scale monitoring

### Performance Optimization

- Use approximate clustering for large datasets
- Implement caching for repeated calculations
- Parallelize metric computations where possible

## API Reference

### ModeCollapseDetector

#### Methods

- `detect_mode_collapse(original_data, synthetic_data, training_metrics=None)`
- `monitor_training(epoch, generator_loss, discriminator_loss, synthetic_data)`
- `get_prevention_strategies(severity)`
- `should_trigger_retraining(recent_metrics, threshold_epochs=5)`
- `generate_visualization(synthetic_data, save_path=None)`
- `get_summary_report()`

### SyntheticDataAgent Integration

#### Methods

- `detect_mode_collapse(original_data, synthetic_data, training_metrics=None)`
- `monitor_training_progress(epoch, generator_loss, discriminator_loss, synthetic_data)`
- `get_mode_collapse_prevention_strategies(severity)`
- `should_trigger_retraining(recent_metrics, threshold_epochs=5)`
- `generate_mode_collapse_visualization(synthetic_data, save_path=None)`
- `get_mode_collapse_summary()`

## Examples

See `test_mode_collapse_detection.py` for comprehensive examples covering:

- Basic mode collapse detection
- Training progress monitoring
- Prevention strategy testing
- Retraining trigger logic
- Visualization generation
- Integration with SyntheticDataAgent
- Comprehensive scenario testing

## Contributing

When contributing to the mode collapse detection system:

1. Add new metrics with proper documentation
2. Include unit tests for new features
3. Update thresholds based on empirical validation
4. Consider computational efficiency for large-scale deployments
5. Maintain backward compatibility with existing APIs 