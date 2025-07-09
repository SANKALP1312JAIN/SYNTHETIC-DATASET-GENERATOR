import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataProfiler:
    def __init__(self):
        # Set style for plots
        plt.style.use('default')
        sns.set_palette("husl")
        
    def __call__(self, state):
        data = state.get('data')
        profile = {}
        
        # Simple data type detection
        if isinstance(data, pd.DataFrame):
            data_type = 'tabular'
            profile['columns'] = list(data.columns)
            profile['shape'] = data.shape
            profile['missing_values'] = data.isnull().sum().to_dict()
            
            # Detect data characteristics
            profile['data_characteristics'] = self._analyze_tabular_data(data)
            
            # Generate quality summary
            profile['quality_summary'] = self._generate_quality_summary(data, state)
            
        elif isinstance(data, list) and all(hasattr(img, 'shape') for img in data):
            data_type = 'image'
            profile['num_images'] = len(data)
            profile['image_shapes'] = [img.shape for img in data]
            profile['data_characteristics'] = self._analyze_image_data(data)
            profile['quality_summary'] = self._generate_image_quality_summary(data, state)
            
        elif isinstance(data, pd.Series) and hasattr(data.index, 'freq'):
            data_type = 'time_series'
            profile['length'] = len(data)
            profile['data_characteristics'] = self._analyze_time_series_data(data)
            profile['quality_summary'] = self._generate_time_series_quality_summary(data, state)
            
        else:
            data_type = 'unknown'
            profile['data_characteristics'] = {}
            profile['quality_summary'] = {}
        
        state['data_type'] = data_type
        state['profile'] = profile
        state['log'] = state.get('log', []) + [f"Data profiled as: {data_type}"]
        
        return state
    
    def _analyze_tabular_data(self, data: pd.DataFrame) -> dict:
        """Analyze tabular data characteristics"""
        characteristics = {}
        
        # Column types
        numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        boolean_cols = data.select_dtypes(include=['bool']).columns.tolist()
        
        characteristics['numerical_columns'] = numerical_cols
        characteristics['categorical_columns'] = categorical_cols
        characteristics['boolean_columns'] = boolean_cols
        
        # Data quality metrics
        characteristics['missing_ratio'] = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        characteristics['duplicate_ratio'] = data.duplicated().sum() / len(data)
        
        # Class imbalance detection for categorical columns
        class_imbalance = False
        imbalance_details = {}
        
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            if len(value_counts) > 1:
                max_count = value_counts.max()
                min_count = value_counts.min()
                imbalance_ratio = min_count / max_count
                
                if imbalance_ratio < 0.1:  # Severe imbalance
                    class_imbalance = True
                    # Fix the minority classes handling - use a more robust approach
                    minority_indices = value_counts.index[1:]
                    # Convert minority indices to list safely
                    try:
                        minority_classes = list(minority_indices)  # type: ignore
                    except (TypeError, AttributeError):
                        minority_classes = [str(minority_indices)]
                    
                    imbalance_details[col] = {
                        'ratio': imbalance_ratio,
                        'majority_class': value_counts.index[0],
                        'minority_classes': minority_classes
                    }
        
        characteristics['class_imbalance'] = class_imbalance
        characteristics['imbalance_details'] = imbalance_details
        
        # Numerical data statistics
        if numerical_cols:
            characteristics['numerical_stats'] = data[numerical_cols].describe().to_dict()
            
            # Detect outliers (simplified approach)
            outlier_columns = []
            for col in numerical_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                if outlier_count > 0:
                    outlier_columns.append(col)
            
            characteristics['outlier_columns'] = outlier_columns
        
        # Data size categories
        if len(data) < 1000:
            characteristics['size_category'] = 'small'
        elif len(data) < 10000:
            characteristics['size_category'] = 'medium'
        else:
            characteristics['size_category'] = 'large'
        
        return characteristics
    
    def _analyze_image_data(self, data: list) -> dict:
        """Analyze image data characteristics"""
        characteristics = {}
        
        # Image dimensions
        heights = [img.shape[0] for img in data]
        widths = [img.shape[1] for img in data]
        channels = [img.shape[2] if len(img.shape) > 2 else 1 for img in data]
        
        characteristics['avg_height'] = np.mean(heights)
        characteristics['avg_width'] = np.mean(widths)
        characteristics['avg_channels'] = np.mean(channels)
        characteristics['consistent_dimensions'] = len(set(heights)) == 1 and len(set(widths)) == 1
        
        # Image quality indicators (simplified)
        characteristics['size_category'] = 'large' if len(data) > 1000 else 'medium' if len(data) > 100 else 'small'
        
        return characteristics
    
    def _analyze_time_series_data(self, data: pd.Series) -> dict:
        """Analyze time series data characteristics"""
        characteristics = {}
        
        # Basic statistics
        characteristics['length'] = len(data)
        characteristics['frequency'] = 'unknown'  # Simplified frequency detection
        characteristics['has_missing_values'] = data.isnull().any()
        
        # Seasonality and trend detection (simplified)
        if len(data) > 10:
            # Simple trend detection
            x = np.arange(len(data))
            slope = np.polyfit(x, data.dropna(), 1)[0]
            characteristics['trend'] = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        
        characteristics['size_category'] = 'large' if len(data) > 1000 else 'medium' if len(data) > 100 else 'small'
        
        return characteristics
    
    def _generate_quality_summary(self, data: pd.DataFrame, state: dict) -> dict:
        """Generate comprehensive data quality summary for tabular data"""
        privacy_level = state.get('config', {}).get('privacy_level', 'medium')
        domain = state.get('domain', 'unknown')
        
        summary = {
            'missing_value_analysis': self._analyze_missing_values(data, privacy_level),
            'class_imbalance_metrics': self._analyze_class_imbalance(data, privacy_level),
            'distribution_analysis': self._analyze_distributions(data, privacy_level),
            'data_quality_issues': [],
            'preprocessing_suggestions': []
        }
        
        # Detect and report issues
        issues = self._detect_data_quality_issues(data, domain)
        summary['data_quality_issues'] = issues
        
        # Generate preprocessing suggestions
        suggestions = self._generate_preprocessing_suggestions(data, issues, domain)
        summary['preprocessing_suggestions'] = suggestions
        
        return summary
    
    def _generate_image_quality_summary(self, data: list, state: dict) -> dict:
        """Generate quality summary for image data"""
        privacy_level = state.get('config', {}).get('privacy_level', 'medium')
        
        summary = {
            'image_resolution_stats': self._analyze_image_resolutions(data, privacy_level),
            'data_quality_issues': [],
            'preprocessing_suggestions': []
        }
        
        # Detect image-specific issues
        issues = self._detect_image_quality_issues(data)
        summary['data_quality_issues'] = issues
        
        # Generate preprocessing suggestions
        suggestions = self._generate_image_preprocessing_suggestions(data, issues)
        summary['preprocessing_suggestions'] = suggestions
        
        return summary
    
    def _generate_time_series_quality_summary(self, data: pd.Series, state: dict) -> dict:
        """Generate quality summary for time series data"""
        privacy_level = state.get('config', {}).get('privacy_level', 'medium')
        
        summary = {
            'temporal_analysis': self._analyze_temporal_patterns(data, privacy_level),
            'data_quality_issues': [],
            'preprocessing_suggestions': []
        }
        
        # Detect time series-specific issues
        issues = self._detect_time_series_quality_issues(data)
        summary['data_quality_issues'] = issues
        
        # Generate preprocessing suggestions
        suggestions = self._generate_time_series_preprocessing_suggestions(data, issues)
        summary['preprocessing_suggestions'] = suggestions
        
        return summary
    
    def _analyze_missing_values(self, data: pd.DataFrame, privacy_level: str) -> dict:
        """Analyze missing values and create heatmap if privacy permits"""
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / len(data)) * 100
        
        analysis = {
            'total_missing': missing_counts.sum(),
            'missing_percentage': (missing_counts.sum() / (len(data) * len(data.columns))) * 100,
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_percentages': missing_percentages[missing_percentages > 0].to_dict()
        }
        
        # Generate heatmap if privacy level allows
        if privacy_level in ['low', 'medium'] and len(data) > 0:
            try:
                plt.figure(figsize=(10, 6))
                sns.heatmap(data.isnull(), yticklabels=False, cbar=True, cmap='viridis')
                plt.title('Missing Values Heatmap')
                plt.tight_layout()
                plt.savefig('missing_values_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                analysis['heatmap_generated'] = True
                analysis['heatmap_path'] = 'missing_values_heatmap.png'
            except Exception as e:
                analysis['heatmap_generated'] = False
                analysis['heatmap_error'] = str(e)
        else:
            analysis['heatmap_generated'] = False
            analysis['reason'] = 'Privacy level too high or insufficient data'
        
        return analysis
    
    def _analyze_class_imbalance(self, data: pd.DataFrame, privacy_level: str) -> dict:
        """Analyze class imbalance in categorical columns"""
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        imbalance_metrics = {}
        
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            if len(value_counts) > 1:
                max_count = value_counts.max()
                min_count = value_counts.min()
                imbalance_ratio = min_count / max_count
                
                metrics = {
                    'imbalance_ratio': imbalance_ratio,
                    'majority_class': value_counts.index[0],
                    'majority_count': max_count,
                    'minority_class': value_counts.index[-1],
                    'minority_count': min_count,
                    'total_classes': len(value_counts),
                    'severity': 'severe' if imbalance_ratio < 0.1 else 'moderate' if imbalance_ratio < 0.3 else 'mild'
                }
                
                # Generate distribution plot if privacy allows
                if privacy_level in ['low', 'medium']:
                    try:
                        plt.figure(figsize=(8, 6))
                        value_counts.plot(kind='bar')
                        plt.title(f'Class Distribution - {col}')
                        plt.xlabel('Classes')
                        plt.ylabel('Count')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig(f'class_distribution_{col}.png', dpi=300, bbox_inches='tight')
                        plt.close()
                        metrics['plot_generated'] = True
                        metrics['plot_path'] = f'class_distribution_{col}.png'
                    except Exception as e:
                        metrics['plot_generated'] = False
                        metrics['plot_error'] = str(e)
                else:
                    metrics['plot_generated'] = False
                    metrics['reason'] = 'Privacy level too high'
                
                imbalance_metrics[col] = metrics
        
        return imbalance_metrics
    
    def _analyze_distributions(self, data: pd.DataFrame, privacy_level: str) -> dict:
        """Analyze column-wise distributions"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        distribution_analysis = {}
        
        for col in numerical_cols:
            analysis = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'skewness': data[col].skew(),
                'kurtosis': data[col].kurtosis()
            }
            
            # Generate distribution plot if privacy allows
            if privacy_level in ['low', 'medium']:
                try:
                    plt.figure(figsize=(8, 6))
                    plt.subplot(2, 1, 1)
                    data[col].hist(bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.title(f'Distribution of {col}')
                    plt.xlabel(col)
                    plt.ylabel('Frequency')
                    
                    plt.subplot(2, 1, 2)
                    data[col].plot(kind='box')
                    plt.title(f'Box Plot of {col}')
                    plt.tight_layout()
                    plt.savefig(f'distribution_{col}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    analysis['plot_generated'] = True
                    analysis['plot_path'] = f'distribution_{col}.png'
                except Exception as e:
                    analysis['plot_generated'] = False
                    analysis['plot_error'] = str(e)
            else:
                analysis['plot_generated'] = False
                analysis['reason'] = 'Privacy level too high'
            
            distribution_analysis[col] = analysis
        
        return distribution_analysis
    
    def _analyze_image_resolutions(self, data: list, privacy_level: str) -> dict:
        """Analyze image resolution statistics"""
        heights = [img.shape[0] for img in data]
        widths = [img.shape[1] for img in data]
        channels = [img.shape[2] if len(img.shape) > 2 else 1 for img in data]
        
        analysis = {
            'total_images': len(data),
            'height_stats': {
                'mean': np.mean(heights),
                'std': np.std(heights),
                'min': np.min(heights),
                'max': np.max(heights)
            },
            'width_stats': {
                'mean': np.mean(widths),
                'std': np.std(widths),
                'min': np.min(widths),
                'max': np.max(widths)
            },
            'channel_stats': {
                'mean': np.mean(channels),
                'std': np.std(channels),
                'min': np.min(channels),
                'max': np.max(channels)
            },
            'consistent_dimensions': len(set(heights)) == 1 and len(set(widths)) == 1
        }
        
        # Generate resolution distribution plot if privacy allows
        if privacy_level in ['low', 'medium'] and len(data) > 0:
            try:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 3, 1)
                plt.hist(heights, bins=20, alpha=0.7, color='red')
                plt.title('Height Distribution')
                plt.xlabel('Height (pixels)')
                plt.ylabel('Frequency')
                
                plt.subplot(1, 3, 2)
                plt.hist(widths, bins=20, alpha=0.7, color='green')
                plt.title('Width Distribution')
                plt.xlabel('Width (pixels)')
                plt.ylabel('Frequency')
                
                plt.subplot(1, 3, 3)
                plt.hist(channels, bins=10, alpha=0.7, color='blue')
                plt.title('Channel Distribution')
                plt.xlabel('Channels')
                plt.ylabel('Frequency')
                
                plt.tight_layout()
                plt.savefig('image_resolution_stats.png', dpi=300, bbox_inches='tight')
                plt.close()
                analysis['plot_generated'] = True
                analysis['plot_path'] = 'image_resolution_stats.png'
            except Exception as e:
                analysis['plot_generated'] = False
                analysis['plot_error'] = str(e)
        else:
            analysis['plot_generated'] = False
            analysis['reason'] = 'Privacy level too high or insufficient data'
        
        return analysis
    
    def _analyze_temporal_patterns(self, data: pd.Series, privacy_level: str) -> dict:
        """Analyze temporal patterns in time series data"""
        analysis = {
            'length': len(data),
            'has_missing_values': data.isnull().any(),
            'missing_count': data.isnull().sum(),
            'missing_percentage': (data.isnull().sum() / len(data)) * 100
        }
        
        # Basic statistics
        if len(data) > 0:
            analysis['basic_stats'] = {
                'mean': data.mean(),
                'std': data.std(),
                'min': data.min(),
                'max': data.max()
            }
        
        # Trend analysis
        if len(data) > 10:
            x = np.arange(len(data))
            slope = np.polyfit(x, data.dropna(), 1)[0]
            analysis['trend'] = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
            analysis['trend_slope'] = slope
        
        # Generate time series plot if privacy allows
        if privacy_level in ['low', 'medium'] and len(data) > 0:
            try:
                plt.figure(figsize=(12, 6))
                data.plot()
                plt.title('Time Series Data')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('time_series_plot.png', dpi=300, bbox_inches='tight')
                plt.close()
                analysis['plot_generated'] = True
                analysis['plot_path'] = 'time_series_plot.png'
            except Exception as e:
                analysis['plot_generated'] = False
                analysis['plot_error'] = str(e)
        else:
            analysis['plot_generated'] = False
            analysis['reason'] = 'Privacy level too high or insufficient data'
        
        return analysis
    
    def _detect_data_quality_issues(self, data: pd.DataFrame, domain: str) -> list:
        """Detect data quality issues"""
        issues = []
        
        # Missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_ratio > 0.1:
            issues.append(f"High missing value ratio: {missing_ratio:.2%}")
        elif missing_ratio > 0.05:
            issues.append(f"Moderate missing value ratio: {missing_ratio:.2%}")
        
        # Duplicates
        duplicate_ratio = data.duplicated().sum() / len(data)
        if duplicate_ratio > 0.1:
            issues.append(f"High duplicate ratio: {duplicate_ratio:.2%}")
        
        # Class imbalance
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            if len(value_counts) > 1:
                imbalance_ratio = value_counts.min() / value_counts.max()
                if imbalance_ratio < 0.1:
                    issues.append(f"Severe class imbalance in '{col}': ratio {imbalance_ratio:.3f}")
                elif imbalance_ratio < 0.3:
                    issues.append(f"Moderate class imbalance in '{col}': ratio {imbalance_ratio:.3f}")
        
        # Outliers in numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_ratio = outlier_count / len(data)
            if outlier_ratio > 0.05:
                issues.append(f"High outlier ratio in '{col}': {outlier_ratio:.2%}")
        
        # Domain-specific issues
        if domain == 'healthcare':
            if any('id' in col.lower() for col in data.columns):
                issues.append("Potential PII detected in healthcare data")
        
        return issues
    
    def _detect_image_quality_issues(self, data: list) -> list:
        """Detect image-specific quality issues"""
        issues = []
        
        if len(data) == 0:
            issues.append("No images provided")
            return issues
        
        # Check for consistent dimensions
        heights = [img.shape[0] for img in data]
        widths = [img.shape[1] for img in data]
        
        if len(set(heights)) > 1 or len(set(widths)) > 1:
            issues.append("Inconsistent image dimensions detected")
        
        # Check for very small images
        min_height = min(heights)
        min_width = min(widths)
        if min_height < 32 or min_width < 32:
            issues.append(f"Very small images detected: min size {min_width}x{min_height}")
        
        # Check for very large images
        max_height = max(heights)
        max_width = max(widths)
        if max_height > 2048 or max_width > 2048:
            issues.append(f"Very large images detected: max size {max_width}x{max_height}")
        
        return issues
    
    def _detect_time_series_quality_issues(self, data: pd.Series) -> list:
        """Detect time series-specific quality issues"""
        issues = []
        
        if len(data) == 0:
            issues.append("No time series data provided")
            return issues
        
        # Missing values
        missing_count = data.isnull().sum()
        if missing_count > 0:
            issues.append(f"Missing values detected: {missing_count} points")
        
        # Check for sufficient length
        if len(data) < 10:
            issues.append("Time series too short for meaningful analysis")
        
        # Check for constant values
        if data.std() == 0:
            issues.append("Time series has no variation (constant values)")
        
        return issues
    
    def _generate_preprocessing_suggestions(self, data: pd.DataFrame, issues: list, domain: str) -> list:
        """Generate preprocessing suggestions based on detected issues"""
        suggestions = []
        
        for issue in issues:
            if "missing value" in issue.lower():
                suggestions.append("Consider imputation strategies: mean, median, or forward fill")
                suggestions.append("Remove rows/columns with high missing ratios if appropriate")
            
            if "duplicate" in issue.lower():
                suggestions.append("Remove duplicate rows to improve data quality")
            
            if "class imbalance" in issue.lower():
                suggestions.append("Consider SMOTE or other oversampling techniques")
                suggestions.append("Use weighted loss functions in models")
            
            if "outlier" in issue.lower():
                suggestions.append("Consider outlier removal or robust scaling")
                suggestions.append("Use IQR or z-score methods for outlier detection")
        
        # Domain-specific suggestions
        if domain == 'healthcare':
            suggestions.append("Ensure HIPAA compliance in data handling")
            suggestions.append("Consider anonymization techniques for sensitive data")
        elif domain == 'finance':
            suggestions.append("Ensure data integrity for financial calculations")
            suggestions.append("Consider robust scaling for financial metrics")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _generate_image_preprocessing_suggestions(self, data: list, issues: list) -> list:
        """Generate preprocessing suggestions for image data"""
        suggestions = []
        
        for issue in issues:
            if "inconsistent dimensions" in issue.lower():
                suggestions.append("Resize all images to consistent dimensions")
                suggestions.append("Use padding or cropping to standardize sizes")
            
            if "small images" in issue.lower():
                suggestions.append("Consider upscaling small images")
                suggestions.append("Use data augmentation to increase dataset size")
            
            if "large images" in issue.lower():
                suggestions.append("Consider downscaling large images for efficiency")
                suggestions.append("Use progressive resizing for training")
        
        # General image suggestions
        suggestions.append("Normalize pixel values to [0,1] or [-1,1] range")
        suggestions.append("Consider data augmentation: rotation, flipping, brightness adjustment")
        
        return list(set(suggestions))
    
    def _generate_time_series_preprocessing_suggestions(self, data: pd.Series, issues: list) -> list:
        """Generate preprocessing suggestions for time series data"""
        suggestions = []
        
        for issue in issues:
            if "missing values" in issue.lower():
                suggestions.append("Use forward fill, backward fill, or interpolation")
                suggestions.append("Consider seasonal decomposition for missing value imputation")
            
            if "too short" in issue.lower():
                suggestions.append("Collect more data points for meaningful analysis")
                suggestions.append("Consider sliding window approaches")
            
            if "no variation" in issue.lower():
                suggestions.append("Check data collection process")
                suggestions.append("Consider differencing to create variation")
        
        # General time series suggestions
        suggestions.append("Consider seasonal decomposition")
        suggestions.append("Apply smoothing techniques if needed")
        suggestions.append("Check for stationarity and apply transformations if necessary")
        
        return list(set(suggestions))
