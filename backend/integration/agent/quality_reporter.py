import pandas as pd
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

class QualityReporter:
    """Generate comprehensive data quality reports"""
    
    def __init__(self):
        self.report_sections = []
        
    def generate_report(self, state: dict) -> str:
        """Generate a comprehensive data quality report"""
        profile = state.get('profile', {})
        quality_summary = profile.get('quality_summary', {})
        data_type = state.get('data_type', 'unknown')
        domain = state.get('domain', 'unknown')
        privacy_level = state.get('config', {}).get('privacy_level', 'medium')
        
        report = []
        report.append("=" * 80)
        report.append("DATA QUALITY SUMMARY REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Domain: {domain}")
        report.append(f"Data Type: {data_type}")
        report.append(f"Privacy Level: {privacy_level}")
        report.append("")
        
        # Basic data information
        report.extend(self._generate_basic_info(profile))
        
        # Data quality analysis based on type
        if data_type == 'tabular':
            report.extend(self._generate_tabular_quality_report(quality_summary, privacy_level))
        elif data_type == 'image':
            report.extend(self._generate_image_quality_report(quality_summary, privacy_level))
        elif data_type == 'time_series':
            report.extend(self._generate_time_series_quality_report(quality_summary, privacy_level))
        else:
            report.append("Data type not supported for detailed quality analysis.")
        
        # Data quality issues
        report.extend(self._generate_issues_report(quality_summary))
        
        # Preprocessing suggestions
        report.extend(self._generate_suggestions_report(quality_summary))
        
        # Privacy considerations
        report.extend(self._generate_privacy_report(privacy_level, domain))
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _generate_basic_info(self, profile: dict) -> List[str]:
        """Generate basic data information section"""
        lines = []
        lines.append("1. BASIC DATA INFORMATION")
        lines.append("-" * 40)
        
        if 'shape' in profile:
            lines.append(f"Dataset Shape: {profile['shape']}")
            lines.append(f"Total Records: {profile['shape'][0]:,}")
            lines.append(f"Total Features: {profile['shape'][1]}")
        
        if 'columns' in profile:
            lines.append(f"Columns: {', '.join(profile['columns'])}")
        
        if 'data_characteristics' in profile:
            chars = profile['data_characteristics']
            if 'size_category' in chars:
                lines.append(f"Dataset Size: {chars['size_category'].title()}")
            
            if 'numerical_columns' in chars:
                lines.append(f"Numerical Columns: {len(chars['numerical_columns'])}")
            
            if 'categorical_columns' in chars:
                lines.append(f"Categorical Columns: {len(chars['categorical_columns'])}")
        
        lines.append("")
        return lines
    
    def _generate_tabular_quality_report(self, quality_summary: dict, privacy_level: str) -> List[str]:
        """Generate quality report for tabular data"""
        lines = []
        lines.append("2. TABULAR DATA QUALITY ANALYSIS")
        lines.append("-" * 40)
        
        # Missing values analysis
        if 'missing_value_analysis' in quality_summary:
            missing_analysis = quality_summary['missing_value_analysis']
            lines.append("Missing Values Analysis:")
            lines.append(f"  Total Missing Values: {missing_analysis.get('total_missing', 0):,}")
            lines.append(f"  Overall Missing Percentage: {missing_analysis.get('missing_percentage', 0):.2f}%")
            
            if missing_analysis.get('heatmap_generated'):
                lines.append(f"  Missing Values Heatmap: {missing_analysis.get('heatmap_path', 'N/A')}")
            elif 'reason' in missing_analysis:
                lines.append(f"  Heatmap: {missing_analysis['reason']}")
            
            if missing_analysis.get('columns_with_missing'):
                lines.append("  Columns with Missing Values:")
                for col, count in missing_analysis['columns_with_missing'].items():
                    percentage = missing_analysis['missing_percentages'].get(col, 0)
                    lines.append(f"    - {col}: {count} ({percentage:.2f}%)")
        
        # Class imbalance analysis
        if 'class_imbalance_metrics' in quality_summary:
            imbalance_metrics = quality_summary['class_imbalance_metrics']
            if imbalance_metrics:
                lines.append("\nClass Imbalance Analysis:")
                for col, metrics in imbalance_metrics.items():
                    lines.append(f"  {col}:")
                    lines.append(f"    Imbalance Ratio: {metrics['imbalance_ratio']:.3f}")
                    lines.append(f"    Severity: {metrics['severity'].title()}")
                    lines.append(f"    Majority Class: {metrics['majority_class']} ({metrics['majority_count']})")
                    lines.append(f"    Minority Class: {metrics['minority_class']} ({metrics['minority_count']})")
                    
                    if metrics.get('plot_generated'):
                        lines.append(f"    Distribution Plot: {metrics.get('plot_path', 'N/A')}")
                    elif 'reason' in metrics:
                        lines.append(f"    Plot: {metrics['reason']}")
        
        # Distribution analysis
        if 'distribution_analysis' in quality_summary:
            dist_analysis = quality_summary['distribution_analysis']
            if dist_analysis:
                lines.append("\nDistribution Analysis:")
                for col, analysis in dist_analysis.items():
                    lines.append(f"  {col}:")
                    lines.append(f"    Mean: {analysis['mean']:.3f}")
                    lines.append(f"    Std: {analysis['std']:.3f}")
                    lines.append(f"    Skewness: {analysis['skewness']:.3f}")
                    lines.append(f"    Kurtosis: {analysis['kurtosis']:.3f}")
                    
                    if analysis.get('plot_generated'):
                        lines.append(f"    Distribution Plot: {analysis.get('plot_path', 'N/A')}")
                    elif 'reason' in analysis:
                        lines.append(f"    Plot: {analysis['reason']}")
        
        lines.append("")
        return lines
    
    def _generate_image_quality_report(self, quality_summary: dict, privacy_level: str) -> List[str]:
        """Generate quality report for image data"""
        lines = []
        lines.append("2. IMAGE DATA QUALITY ANALYSIS")
        lines.append("-" * 40)
        
        if 'image_resolution_stats' in quality_summary:
            resolution_stats = quality_summary['image_resolution_stats']
            lines.append("Image Resolution Statistics:")
            lines.append(f"  Total Images: {resolution_stats.get('total_images', 0):,}")
            
            height_stats = resolution_stats.get('height_stats', {})
            if height_stats:
                lines.append("  Height Statistics:")
                lines.append(f"    Mean: {height_stats.get('mean', 0):.1f} pixels")
                lines.append(f"    Std: {height_stats.get('std', 0):.1f} pixels")
                lines.append(f"    Range: {height_stats.get('min', 0)} - {height_stats.get('max', 0)} pixels")
            
            width_stats = resolution_stats.get('width_stats', {})
            if width_stats:
                lines.append("  Width Statistics:")
                lines.append(f"    Mean: {width_stats.get('mean', 0):.1f} pixels")
                lines.append(f"    Std: {width_stats.get('std', 0):.1f} pixels")
                lines.append(f"    Range: {width_stats.get('min', 0)} - {width_stats.get('max', 0)} pixels")
            
            channel_stats = resolution_stats.get('channel_stats', {})
            if channel_stats:
                lines.append("  Channel Statistics:")
                lines.append(f"    Mean: {channel_stats.get('mean', 0):.1f} channels")
                lines.append(f"    Range: {channel_stats.get('min', 0)} - {channel_stats.get('max', 0)} channels")
            
            lines.append(f"  Consistent Dimensions: {resolution_stats.get('consistent_dimensions', False)}")
            
            if resolution_stats.get('plot_generated'):
                lines.append(f"  Resolution Distribution Plot: {resolution_stats.get('plot_path', 'N/A')}")
            elif 'reason' in resolution_stats:
                lines.append(f"  Plot: {resolution_stats['reason']}")
        
        lines.append("")
        return lines
    
    def _generate_time_series_quality_report(self, quality_summary: dict, privacy_level: str) -> List[str]:
        """Generate quality report for time series data"""
        lines = []
        lines.append("2. TIME SERIES DATA QUALITY ANALYSIS")
        lines.append("-" * 40)
        
        if 'temporal_analysis' in quality_summary:
            temporal_analysis = quality_summary['temporal_analysis']
            lines.append("Temporal Analysis:")
            lines.append(f"  Series Length: {temporal_analysis.get('length', 0):,} points")
            lines.append(f"  Missing Values: {temporal_analysis.get('missing_count', 0)} ({temporal_analysis.get('missing_percentage', 0):.2f}%)")
            
            basic_stats = temporal_analysis.get('basic_stats', {})
            if basic_stats:
                lines.append("  Basic Statistics:")
                lines.append(f"    Mean: {basic_stats.get('mean', 0):.3f}")
                lines.append(f"    Std: {basic_stats.get('std', 0):.3f}")
                lines.append(f"    Range: {basic_stats.get('min', 0):.3f} - {basic_stats.get('max', 0):.3f}")
            
            if 'trend' in temporal_analysis:
                lines.append(f"  Trend: {temporal_analysis['trend'].title()}")
                lines.append(f"  Trend Slope: {temporal_analysis.get('trend_slope', 0):.6f}")
            
            if temporal_analysis.get('plot_generated'):
                lines.append(f"  Time Series Plot: {temporal_analysis.get('plot_path', 'N/A')}")
            elif 'reason' in temporal_analysis:
                lines.append(f"  Plot: {temporal_analysis['reason']}")
        
        lines.append("")
        return lines
    
    def _generate_issues_report(self, quality_summary: dict) -> List[str]:
        """Generate data quality issues report"""
        lines = []
        lines.append("3. DATA QUALITY ISSUES")
        lines.append("-" * 40)
        
        issues = quality_summary.get('data_quality_issues', [])
        if issues:
            lines.append("Detected Issues:")
            for i, issue in enumerate(issues, 1):
                lines.append(f"  {i}. {issue}")
        else:
            lines.append("No significant data quality issues detected.")
        
        lines.append("")
        return lines
    
    def _generate_suggestions_report(self, quality_summary: dict) -> List[str]:
        """Generate preprocessing suggestions report"""
        lines = []
        lines.append("4. PREPROCESSING SUGGESTIONS")
        lines.append("-" * 40)
        
        suggestions = quality_summary.get('preprocessing_suggestions', [])
        if suggestions:
            lines.append("Recommended Preprocessing Steps:")
            for i, suggestion in enumerate(suggestions, 1):
                lines.append(f"  {i}. {suggestion}")
        else:
            lines.append("No specific preprocessing suggestions at this time.")
        
        lines.append("")
        return lines
    
    def _generate_privacy_report(self, privacy_level: str, domain: str) -> List[str]:
        """Generate privacy considerations report"""
        lines = []
        lines.append("5. PRIVACY CONSIDERATIONS")
        lines.append("-" * 40)
        
        lines.append(f"Privacy Level: {privacy_level.title()}")
        lines.append(f"Domain: {domain.title()}")
        
        if privacy_level == 'high':
            lines.append("  - High privacy requirements detected")
            lines.append("  - Visualizations may be limited")
            lines.append("  - Consider additional anonymization techniques")
        elif privacy_level == 'medium':
            lines.append("  - Medium privacy requirements")
            lines.append("  - Basic visualizations allowed")
            lines.append("  - Ensure data anonymization")
        else:
            lines.append("  - Low privacy requirements")
            lines.append("  - Full visualizations available")
            lines.append("  - Standard data handling practices")
        
        # Domain-specific privacy notes
        if domain == 'healthcare':
            lines.append("  - HIPAA compliance required")
            lines.append("  - PHI must be properly anonymized")
        elif domain == 'finance':
            lines.append("  - Financial data regulations apply")
            lines.append("  - Ensure data integrity")
        
        lines.append("")
        return lines
    
    def save_report(self, report: str, filename: Optional[str] = None) -> str:
        """Save the report to a file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data_quality_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return filename
    
    def generate_json_summary(self, state: dict) -> dict:
        """Generate a JSON summary of the quality analysis"""
        profile = state.get('profile', {})
        quality_summary = profile.get('quality_summary', {})
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'domain': state.get('domain', 'unknown'),
            'data_type': state.get('data_type', 'unknown'),
            'privacy_level': state.get('config', {}).get('privacy_level', 'medium'),
            'basic_info': {
                'shape': profile.get('shape'),
                'columns': profile.get('columns'),
                'data_characteristics': profile.get('data_characteristics', {})
            },
            'quality_metrics': {
                'missing_value_analysis': quality_summary.get('missing_value_analysis', {}),
                'class_imbalance_metrics': quality_summary.get('class_imbalance_metrics', {}),
                'distribution_analysis': quality_summary.get('distribution_analysis', {}),
                'image_resolution_stats': quality_summary.get('image_resolution_stats', {}),
                'temporal_analysis': quality_summary.get('temporal_analysis', {})
            },
            'issues': quality_summary.get('data_quality_issues', []),
            'suggestions': quality_summary.get('preprocessing_suggestions', [])
        }
        
        return summary 