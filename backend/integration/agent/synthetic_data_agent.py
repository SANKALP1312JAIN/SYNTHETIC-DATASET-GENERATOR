from langgraph.graph import StateGraph, END
from .domain_analyzer import DomainAnalyzer
from .data_profiler import DataProfiler
from .data_preprocessor import DataPreprocessor
from .gan_selector import GANSelector
from .quality_evaluator import QualityEvaluator
from .privacy_guardian import PrivacyGuardian
from .feedback_agent import FeedbackAgent
from .quality_reporter import QualityReporter
from .synthetic_data_generator import SyntheticDataGenerator
from .mode_collapse_detector import ModeCollapseDetector, mode_collapse_detector
from pydantic import BaseModel
from typing import Optional, Union, Any, List, Dict
from pathlib import Path
from datetime import datetime

class AgentState(BaseModel):
    config: dict
    data: object  # or a more specific type, e.g., pd.DataFrame
    log: list = []
    user_context: Optional[dict] = None
    privacy_checks: Optional[dict] = None
    fairness_checks: Optional[dict] = None
    compliance_checks: Optional[dict] = None

class SyntheticDataAgent:
    def __init__(self):
        self.domain_analyzer = DomainAnalyzer()
        self.data_profiler = DataProfiler()
        self.data_preprocessor = DataPreprocessor()
        self.gan_selector = GANSelector()
        self.quality_evaluator = QualityEvaluator()
        self.privacy_guardian = PrivacyGuardian()
        self.feedback_agent = FeedbackAgent()
        self.quality_reporter = QualityReporter()
        self.synthetic_data_generator = SyntheticDataGenerator()
        self.mode_collapse_detector = mode_collapse_detector

        self.graph = StateGraph(AgentState)
        # Register sub-agents as nodes
        self.graph.add_node('domain_analyzer', self.domain_analyzer)
        self.graph.add_node('data_profiler', self.data_profiler)
        self.graph.add_node('gan_selector', self.gan_selector)
        self.graph.add_node('quality_evaluator', self.quality_evaluator)
        self.graph.add_node('privacy_guardian', self.privacy_guardian)
        self.graph.add_node('feedback_agent', self.feedback_agent)
        # Define basic flow (customize as needed)
        self.graph.add_edge('domain_analyzer', 'data_profiler')
        self.graph.add_edge('data_profiler', 'gan_selector')
        self.graph.add_edge('gan_selector', 'quality_evaluator')
        self.graph.add_edge('quality_evaluator', 'privacy_guardian')
        self.graph.add_edge('privacy_guardian', 'feedback_agent')
        self.graph.add_edge('feedback_agent', END)

    def run(self, initial_state):
        """
        Run the LANGgraph pipeline starting from the initial state.
        """
        compiled_graph = self.graph.compile()
        return compiled_graph.invoke(initial_state)
    
    def preprocess_and_run(self, data_source, config: dict):
        """
        Preprocess data and run the synthetic data generation pipeline
        
        Args:
            data_source: File path, DataFrame, or list of data
            config: Configuration dictionary
            
        Returns:
            Final state from the pipeline
        """
        # Preprocess the data
        processed_data, detected_type, preprocessing_metadata = self.data_preprocessor.preprocess_data(
            data_source, config.get('data_type')
        )
        
        # Update config with detected type
        config['data_type'] = detected_type
        
        # Create initial state
        initial_state = {
            'config': config,
            'data': processed_data,
            'preprocessing_metadata': preprocessing_metadata
        }
        
        # Run the pipeline
        return self.run(initial_state)
    
    def generate_quality_report(self, state):
        """
        Generate a comprehensive data quality report
        """
        return self.quality_reporter.generate_report(state)
    
    def save_quality_report(self, state, filename: Optional[str] = None):
        """
        Save the quality report to a file
        """
        report = self.generate_quality_report(state)
        return self.quality_reporter.save_report(report, filename)
    
    def get_quality_summary(self, state):
        """
        Get a JSON summary of the quality analysis
        """
        return self.quality_reporter.generate_json_summary(state)
    
    def validate_data_source(self, data_source):
        """
        Validate if a data source can be processed
        """
        if isinstance(data_source, (str, Path)):
            return self.data_preprocessor.validate_file(data_source)
        else:
            return {'valid': True, 'source_type': type(data_source).__name__}
    
    def get_supported_formats(self):
        """
        Get list of supported file formats
        """
        return self.data_preprocessor.get_supported_formats()
    
    def generate_synthetic_data(self, data: Any, gan_type: str, num_samples: int,
                               domain: str = 'general', privacy_level: str = 'medium'):
        """
        Generate synthetic data using selected GAN
        
        Args:
            data: Original data for training
            gan_type: Type of GAN to use
            num_samples: Number of synthetic samples to generate
            domain: Domain of the data
            privacy_level: Privacy level
            
        Returns:
            Tuple of (synthetic_data, generation_metadata)
        """
        return self.synthetic_data_generator.generate_synthetic_data(
            data, gan_type, num_samples, domain, privacy_level
        )
    
    def convert_synthetic_data(self, synthetic_data: Any, output_format: str,
                              output_path: str, domain: str = 'general',
                              privacy_level: str = 'medium'):
        """
        Convert synthetic data to requested output format
        
        Args:
            synthetic_data: Generated synthetic data
            output_format: Desired output format
            output_path: Path to save the output
            domain: Domain of the data
            privacy_level: Privacy level
            
        Returns:
            Dictionary with conversion metadata
        """
        return self.synthetic_data_generator.convert_to_format(
            synthetic_data, output_format, output_path, domain, privacy_level
        )
    
    def get_supported_output_formats(self):
        """
        Get list of supported output formats
        """
        return self.synthetic_data_generator.get_supported_output_formats()

    # Mode Collapse Detection Methods
    
    def detect_mode_collapse(self, original_data, synthetic_data, 
                           training_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Detect mode collapse in synthetic data
        
        Args:
            original_data: Original training data
            synthetic_data: Generated synthetic data
            training_metrics: Optional training metrics
            
        Returns:
            Mode collapse detection results
        """
        result = self.mode_collapse_detector.detect_mode_collapse(
            original_data, synthetic_data, training_metrics
        )
        # Convert dataclass to dict for return type compatibility
        return {
            'timestamp': result.timestamp,
            'diversity_score': result.diversity_score,
            'uniqueness_ratio': result.uniqueness_ratio,
            'cluster_score': result.cluster_score,
            'distribution_similarity': result.distribution_similarity,
            'mode_collapse_detected': result.mode_collapse_detected,
            'severity': result.severity,
            'recommendations': result.recommendations,
            'metadata': result.metadata
        }
    
    def monitor_training_progress(self, epoch: int, generator_loss: float, 
                                discriminator_loss: float, synthetic_data) -> Dict[str, Any]:
        """
        Monitor training progress for mode collapse indicators
        
        Args:
            epoch: Current training epoch
            generator_loss: Generator loss value
            discriminator_loss: Discriminator loss value
            synthetic_data: Current synthetic data
            
        Returns:
            Training monitoring results
        """
        result = self.mode_collapse_detector.monitor_training(
            epoch, generator_loss, discriminator_loss, synthetic_data
        )
        # Convert dataclass to dict for return type compatibility
        return {
            'epoch': result.epoch,
            'generator_loss': result.generator_loss,
            'discriminator_loss': result.discriminator_loss,
            'diversity_score': result.diversity_score,
            'synthetic_samples': result.synthetic_samples,
            'unique_samples': result.unique_samples,
            'mode_collapse_risk': result.mode_collapse_risk
        }
    
    def get_mode_collapse_prevention_strategies(self, severity: str) -> Dict[str, Any]:
        """
        Get prevention strategies based on severity
        
        Args:
            severity: Mode collapse severity level
            
        Returns:
            Prevention strategies
        """
        return self.mode_collapse_detector.get_prevention_strategies(severity)
    
    def should_trigger_retraining(self, recent_metrics: List[Any], 
                                threshold_epochs: int = 5) -> bool:
        """
        Determine if retraining should be triggered
        
        Args:
            recent_metrics: Recent mode collapse metrics
            threshold_epochs: Number of epochs to consider
            
        Returns:
            True if retraining should be triggered
        """
        return self.mode_collapse_detector.should_trigger_retraining(recent_metrics, threshold_epochs)
    
    def generate_mode_collapse_visualization(self, synthetic_data, 
                                           save_path: Optional[str] = None) -> str:
        """
        Generate visualization for mode collapse analysis
        
        Args:
            synthetic_data: Synthetic data to analyze
            save_path: Path to save visualization
            
        Returns:
            Path to saved visualization
        """
        if save_path is None:
            save_path = 'mode_collapse_analysis.png'
        return self.mode_collapse_detector.generate_visualization(synthetic_data, save_path)
    
    def get_mode_collapse_summary(self) -> Dict[str, Any]:
        """
        Get summary report of mode collapse monitoring
        
        Returns:
            Summary report
        """
        return self.mode_collapse_detector.get_summary_report()



if __name__ == "__main__":
    # Simple example usage
    print("SyntheticDataAgent - Basic Usage Example")
    print("Use the CustomDatasetGenerator for interactive usage:")
    print("python custom_dataset_generator.py")
