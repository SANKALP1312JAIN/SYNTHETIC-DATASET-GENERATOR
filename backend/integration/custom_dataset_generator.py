#!/usr/bin/env python3
"""
Custom Dataset Synthetic Data Generator
A flexible interface for uploading any feasible data type and generating synthetic data
with optional metadata and requirements.
"""

import pandas as pd
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import sys

# Add src to path for imports
sys.path.append('src')

class CustomDatasetGenerator:
    """Flexible interface for custom dataset synthetic data generation"""
    
    def __init__(self):
        # Lazy load heavy dependencies only when needed
        self._agent = None
        self._generator = None
        
        # Supported file formats
        self.supported_formats = {
            'tabular': ['.csv', '.xlsx', '.json', '.parquet', '.tsv', '.txt'],
            'image': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'],
            'archive': ['.zip', '.tar', '.gz'],
            'dicom': ['.dcm', '.dicom']
        }
    
    @property
    def agent(self):
        """Lazy load the synthetic data agent"""
        if self._agent is None:
            try:
                from integration.agent.synthetic_data_agent import SyntheticDataAgent
                self._agent = SyntheticDataAgent()
            except ImportError as e:
                print(f"Warning: Could not import SyntheticDataAgent: {e}")
                self._agent = None
        return self._agent
    
    @property
    def generator(self):
        """Lazy load the synthetic data generator"""
        if self._generator is None:
            try:
                from integration.agent.synthetic_data_generator import SyntheticDataGenerator
                self._generator = SyntheticDataGenerator()
            except ImportError as e:
                print(f"Warning: Could not import SyntheticDataGenerator: {e}")
                self._generator = None
        return self._generator
    
    def load_dataset(self, file_path: Union[str, Path]) -> Any:
        """
        Load dataset from file path, automatically detecting type
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Loaded data (DataFrame, list of images, etc.)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        # Tabular data
        if file_extension in self.supported_formats['tabular']:
            if file_extension == '.csv':
                return pd.read_csv(file_path)
            elif file_extension == '.xlsx':
                return pd.read_excel(file_path)
            elif file_extension == '.json':
                return pd.read_json(file_path)
            elif file_extension == '.parquet':
                return pd.read_parquet(file_path)
            elif file_extension in ['.tsv', '.txt']:
                return pd.read_csv(file_path, sep='\t')
        
        # Archive files (ZIP)
        elif file_extension == '.zip':
            return self._load_zip_archive(file_path)
        
        # Image files
        elif file_extension in self.supported_formats['image']:
            return self._load_image_file(file_path)
        
        # DICOM files
        elif file_extension in self.supported_formats['dicom']:
            return self._load_dicom_file(file_path)
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _load_zip_archive(self, file_path: Path) -> Dict[str, Any]:
        """Load data from ZIP archive"""
        import zipfile
        
        data = {}
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                file_ext = Path(file_name).suffix.lower()
                
                if file_ext in self.supported_formats['tabular']:
                    # Load tabular data from ZIP
                    with zip_ref.open(file_name) as f:
                        if file_ext == '.csv':
                            data[file_name] = pd.read_csv(f)
                        elif file_ext == '.xlsx':
                            data[file_name] = pd.read_excel(f)
                        elif file_ext == '.json':
                            data[file_name] = pd.read_json(f)
                
                elif file_ext in self.supported_formats['image']:
                    # Load images from ZIP
                    with zip_ref.open(file_name) as f:
                        try:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(f.read()))
                            data[file_name] = np.array(img)
                        except ImportError:
                            print(f"PIL not available, skipping image: {file_name}")
        
        return data
    
    def _load_image_file(self, file_path: Path) -> np.ndarray:
        """Load single image file"""
        try:
            from PIL import Image
            img = Image.open(file_path)
            return np.array(img)
        except ImportError:
            raise ImportError("PIL/Pillow required for image processing")
    
    def _load_dicom_file(self, file_path: Path) -> Dict[str, Any]:
        """Load DICOM file"""
        try:
            import pydicom
            ds = pydicom.dcmread(file_path)
            return {
                'filename': file_path.name,
                'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                'modality': getattr(ds, 'Modality', 'Unknown'),
                'pixel_data': ds.pixel_array if hasattr(ds, 'pixel_array') else None,
                'dicom_metadata': {elem.name: elem.value for elem in ds}
            }
        except ImportError:
            raise ImportError("pydicom required for DICOM processing. Please install it with 'pip install pydicom'.")
    
    def load_dataset_batch(self, file_path: Union[str, Path], batch_size: int = 10000) -> Any:
        """
        Load large dataset in batches to reduce memory usage
        
        Args:
            file_path: Path to the dataset file
            batch_size: Number of rows to load at once
            
        Returns:
            Generator yielding data batches
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        # Only support tabular data for batch processing
        if file_extension in self.supported_formats['tabular']:
            if file_extension == '.csv':
                # Use pandas chunking for large CSV files
                for chunk in pd.read_csv(file_path, chunksize=batch_size):
                    yield chunk
            elif file_extension == '.xlsx':
                # Excel files need to be loaded entirely due to format limitations
                data = pd.read_excel(file_path)
                for i in range(0, len(data), batch_size):
                    yield data.iloc[i:i + batch_size]
            elif file_extension == '.json':
                # JSON files need to be loaded entirely
                data = pd.read_json(file_path)
                for i in range(0, len(data), batch_size):
                    yield data.iloc[i:i + batch_size]
            else:
                # For other formats, load entirely
                data = self.load_dataset(file_path)
                for i in range(0, len(data), batch_size):
                    yield data.iloc[i:i + batch_size]
        else:
            # For non-tabular data, load entirely
            yield self.load_dataset(file_path)
    
    def generate_synthetic_data(self, 
                               data: Any,
                               metadata: Optional[Dict[str, Any]] = None,
                               requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate synthetic data with custom metadata and requirements
        
        Args:
            data: Input dataset
            metadata: Optional metadata about the dataset
            requirements: Optional requirements for synthetic generation
            
        Returns:
            Dictionary containing synthetic data and generation info
        """
        # Check if generator is available
        if self.generator is None:
            return {
                'success': False,
                'error': 'SyntheticDataGenerator not available. Please install required dependencies.',
                'input_metadata': metadata or {},
                'requirements': requirements or {}
            }
        
        # Set default metadata
        if metadata is None:
            metadata = {}
        
        # Set default requirements
        if requirements is None:
            requirements = {}
        
        # Extract requirements with defaults
        domain = requirements.get('domain', 'general')
        privacy_level = requirements.get('privacy_level', 'medium')
        num_samples = requirements.get('num_samples', 100)
        gan_type = requirements.get('gan_type', 'auto')
        output_format = requirements.get('output_format', 'csv')
        output_path = requirements.get('output_path', 'synthetic_data')
        
        # Auto-select GAN type if not specified
        if gan_type == 'auto':
            if isinstance(data, pd.DataFrame):
                gan_type = 'DP-CTGAN' if privacy_level == 'high' else 'CTGAN'
            elif isinstance(data, list) and hasattr(data[0], 'shape'):
                gan_type = 'MedGAN'
            else:
                gan_type = 'DefaultGAN'
        
        # Generate synthetic data
        try:
            start_time = time.time()
            generator = self.generator
            if generator is None:
                raise RuntimeError("Generator not available")
                
            # Type assertion for the generator
            from integration.agent.synthetic_data_generator import SyntheticDataGenerator
            if not isinstance(generator, SyntheticDataGenerator):
                raise RuntimeError("Invalid generator type")
                
            synthetic_data, generation_metadata = generator.generate_synthetic_data(
                data, gan_type, num_samples, domain, privacy_level
            )
            
            # Convert to requested format
            conversion_metadata = generator.convert_to_format(
                synthetic_data, output_format, output_path, domain, privacy_level
            )
            
            end_time = time.time()
            generation_time = round(end_time - start_time, 2)
            
            return {
                'success': True,
                'synthetic_data': synthetic_data,
                'generation_metadata': generation_metadata,
                'conversion_metadata': conversion_metadata,
                'input_metadata': metadata,
                'requirements': requirements,
                'generation_time': generation_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'input_metadata': metadata,
                'requirements': requirements
            }
    
    def generate_synthetic_data_batch(self, 
                                     file_path: str,
                                     metadata: Optional[Dict[str, Any]] = None,
                                     requirements: Optional[Dict[str, Any]] = None,
                                     batch_size: int = 10000) -> Dict[str, Any]:
        """
        Generate synthetic data for large files using batch processing
        
        Args:
            file_path: Path to the dataset file
            metadata: Optional metadata about the dataset
            requirements: Optional requirements for synthetic generation
            batch_size: Size of batches for processing
            
        Returns:
            Dictionary containing synthetic data and generation info
        """
        # Check if generator is available
        if self.generator is None:
            return {
                'success': False,
                'error': 'SyntheticDataGenerator not available. Please install required dependencies.',
                'input_metadata': metadata or {},
                'requirements': requirements or {}
            }
        
        if metadata is None:
            metadata = {}
        
        if requirements is None:
            requirements = {}
        
        # Extract requirements with defaults
        domain = requirements.get('domain', 'general')
        privacy_level = requirements.get('privacy_level', 'medium')
        num_samples = requirements.get('num_samples', 100)
        gan_type = requirements.get('gan_type', 'auto')
        output_format = requirements.get('output_format', 'csv')
        output_path = requirements.get('output_path', 'synthetic_data')
        
        start_time = time.time()
        all_synthetic_data = []
        
        try:
            # Process data in batches
            batch_count = 0
            for batch in self.load_dataset_batch(file_path, batch_size):
                batch_count += 1
                print(f"Processing batch {batch_count}...")
                
                # Generate synthetic data for this batch
                batch_samples = min(num_samples // batch_count, len(batch))
                if batch_samples <= 0:
                    break
                
                synthetic_batch, _ = self.generator.generate_synthetic_data(
                    batch, gan_type, batch_samples, domain, privacy_level
                )
                
                all_synthetic_data.append(synthetic_batch)
            
            # Combine all batches
            if isinstance(all_synthetic_data[0], pd.DataFrame):
                synthetic_data = pd.concat(all_synthetic_data, ignore_index=True)
            else:
                synthetic_data = [item for sublist in all_synthetic_data for item in sublist]
            
            # Convert to requested format
            conversion_metadata = self.generator.convert_to_format(
                synthetic_data, output_format, output_path, domain, privacy_level
            )
            
            end_time = time.time()
            generation_time = round(end_time - start_time, 2)
            
            return {
                'success': True,
                'synthetic_data': synthetic_data,
                'generation_metadata': {'batches_processed': batch_count},
                'conversion_metadata': conversion_metadata,
                'input_metadata': metadata,
                'requirements': requirements,
                'generation_time': generation_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'input_metadata': metadata,
                'requirements': requirements
            }
    
    def interactive_generation(self, file_path: str) -> Dict[str, Any]:
        """
        Interactive synthetic data generation with user input
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Generation results
        """
        print(f"Loading dataset from: {file_path}")
        data = self.load_dataset(file_path)
        
        print(f"Dataset loaded successfully!")
        print(f"Data type: {type(data).__name__}")
        
        if isinstance(data, pd.DataFrame):
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
        elif isinstance(data, list):
            print(f"Number of items: {len(data)}")
        elif isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
        
        # Get user requirements
        print("\n" + "="*50)
        print("SYNTHETIC DATA GENERATION SETUP")
        print("="*50)
        
        # Domain selection
        print("\nAvailable domains:")
        domains = ['general', 'healthcare', 'finance', 'retail', 'education', 'research']
        for i, domain in enumerate(domains, 1):
            print(f"{i}. {domain}")
        
        domain_choice = input(f"Select domain (1-{len(domains)}, default: general): ").strip()
        domain = domains[int(domain_choice) - 1] if domain_choice.isdigit() and 1 <= int(domain_choice) <= len(domains) else 'general'
        
        # Privacy level
        print("\nPrivacy levels:")
        privacy_levels = ['low', 'medium', 'high']
        for i, level in enumerate(privacy_levels, 1):
            print(f"{i}. {level} - {self._get_privacy_description(level)}")
        
        privacy_choice = input(f"Select privacy level (1-{len(privacy_levels)}, default: medium): ").strip()
        privacy_level = privacy_levels[int(privacy_choice) - 1] if privacy_choice.isdigit() and 1 <= int(privacy_choice) <= len(privacy_levels) else 'medium'
        
        # Number of samples
        num_samples = input(f"Number of synthetic samples to generate (default: 100): ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 100
        
        # GAN type
        print("\nGAN types:")
        gan_types = ['auto', 'DP-CTGAN', 'CTGAN', 'MedGAN', 'DefaultGAN']
        for i, gan in enumerate(gan_types, 1):
            print(f"{i}. {gan}")
        
        gan_choice = input(f"Select GAN type (1-{len(gan_types)}, default: auto): ").strip()
        gan_type = gan_types[int(gan_choice) - 1] if gan_choice.isdigit() and 1 <= int(gan_choice) <= len(gan_types) else 'auto'
        
        # Output format
        print("\nOutput formats:")
        output_formats = ['csv', 'json', 'xlsx', 'zip', 'png', 'dcm']
        for i, fmt in enumerate(output_formats, 1):
            print(f"{i}. {fmt}")
        
        format_choice = input(f"Select output format (1-{len(output_formats)}, default: csv): ").strip()
        output_format = output_formats[int(format_choice) - 1] if format_choice.isdigit() and 1 <= int(format_choice) <= len(output_formats) else 'csv'
        
        # Output path
        output_path = input(f"Output path (default: synthetic_data): ").strip()
        output_path = output_path if output_path else 'synthetic_data'
        
        # Optional metadata
        print("\nOptional metadata (press Enter to skip):")
        description = input("Dataset description: ").strip()
        source = input("Data source: ").strip()
        version = input("Version: ").strip()
        
        # Build requirements and metadata
        requirements = {
            'domain': domain,
            'privacy_level': privacy_level,
            'num_samples': num_samples,
            'gan_type': gan_type,
            'output_format': output_format,
            'output_path': output_path
        }
        
        metadata = {}
        if description:
            metadata['description'] = description
        if source:
            metadata['source'] = source
        if version:
            metadata['version'] = version
        
        # Generate synthetic data
        print(f"\nGenerating synthetic data...")
        result = self.generate_synthetic_data(data, metadata, requirements)
        
        return result
    
    def _get_privacy_description(self, level: str) -> str:
        """Get description for privacy level"""
        descriptions = {
            'low': 'Minimal privacy protection, may include original patterns',
            'medium': 'Moderate privacy protection, some pattern preservation',
            'high': 'High privacy protection, removes sensitive information'
        }
        return descriptions.get(level, 'Unknown privacy level')
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported file formats"""
        return self.supported_formats
    
    def generate_simple_synthetic_data(self, data: Any, num_samples: int = 100) -> Dict[str, Any]:
        """
        Generate simple synthetic data without heavy dependencies
        
        Args:
            data: Input dataset
            num_samples: Number of synthetic samples to generate
            
        Returns:
            Dictionary containing synthetic data
        """
        try:
            if isinstance(data, pd.DataFrame):
                # Simple synthetic generation for tabular data
                synthetic_data = {}
                
                # Process numerical columns
                numerical_cols = data.select_dtypes(include=[np.number]).columns
                for col in numerical_cols:
                    mean_val = data[col].mean()
                    std_val = data[col].std()
                    synthetic_data[col] = np.random.normal(mean_val, std_val, num_samples)
                
                # Process categorical columns
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    categories = data[col].dropna().unique()
                    if len(categories) > 0:
                        synthetic_data[col] = np.random.choice(categories.tolist(), num_samples)
                    else:
                        synthetic_data[col] = ['Unknown'] * num_samples
                
                synthetic_df = pd.DataFrame(synthetic_data)
                
                return {
                    'success': True,
                    'synthetic_data': synthetic_df,
                    'method': 'simple_statistical',
                    'num_samples': num_samples
                }
            else:
                return {
                    'success': False,
                    'error': 'Simple synthetic generation only supports pandas DataFrames'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_performance_recommendations(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get performance recommendations based on file characteristics
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            Dictionary with performance recommendations
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'error': 'File not found'}
        
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        file_extension = file_path.suffix.lower()
        
        recommendations = {
            'file_size_mb': round(file_size_mb, 2),
            'file_type': file_extension,
            'estimated_time': 'Unknown',
            'recommended_approach': 'standard',
            'optimization_tips': []
        }
        
        # Estimate generation time based on file size and type
        if file_extension in self.supported_formats['tabular']:
            if file_size_mb < 1:
                recommendations['estimated_time'] = '5-15 seconds'
                recommendations['recommended_approach'] = 'standard'
            elif file_size_mb < 10:
                recommendations['estimated_time'] = '30 seconds - 2 minutes'
                recommendations['recommended_approach'] = 'standard'
            elif file_size_mb < 100:
                recommendations['estimated_time'] = '2-10 minutes'
                recommendations['recommended_approach'] = 'batch'
                recommendations['optimization_tips'].append('Use batch processing for better memory efficiency')
            else:
                recommendations['estimated_time'] = '10+ minutes'
                recommendations['recommended_approach'] = 'batch'
                recommendations['optimization_tips'].extend([
                    'Use batch processing for better memory efficiency',
                    'Consider reducing the number of synthetic samples',
                    'Use simpler GAN types for faster generation'
                ])
        
        elif file_extension in self.supported_formats['image']:
            if file_size_mb < 10:
                recommendations['estimated_time'] = '1-3 minutes'
            elif file_size_mb < 100:
                recommendations['estimated_time'] = '5-15 minutes'
            else:
                recommendations['estimated_time'] = '15+ minutes'
                recommendations['optimization_tips'].append('Consider processing images in smaller batches')
        
        elif file_extension in self.supported_formats['dicom']:
            if file_size_mb < 50:
                recommendations['estimated_time'] = '2-5 minutes'
            elif file_size_mb < 200:
                recommendations['estimated_time'] = '5-15 minutes'
            else:
                recommendations['estimated_time'] = '15+ minutes'
        
        # Add general optimization tips
        if file_size_mb > 50:
            recommendations['optimization_tips'].extend([
                'Close other applications to free up memory',
                'Use SSD storage for faster I/O operations',
                'Consider using a machine with more RAM'
            ])
        
        return recommendations

def main():
    """Main function for command-line usage"""
    print("CUSTOM DATASET SYNTHETIC DATA GENERATOR")
    print("="*60)
    
    # Initialize generator (this should be fast now)
    print("Initializing...")
    generator = CustomDatasetGenerator()
    print("✓ Initialized successfully!")
    
    # Get file path
    file_path = input("Enter the path to your dataset file: ").strip()
    
    if not file_path:
        print("No file path provided. Exiting.")
        return
    
    try:
        # Get performance recommendations
        print("\n" + "="*60)
        print("PERFORMANCE ANALYSIS")
        print("="*60)
        recommendations = generator.get_performance_recommendations(file_path)
        
        if 'error' not in recommendations:
            print(f"File size: {recommendations['file_size_mb']} MB")
            print(f"File type: {recommendations['file_type']}")
            print(f"Estimated generation time: {recommendations['estimated_time']}")
            print(f"Recommended approach: {recommendations['recommended_approach']}")
            
            if recommendations['optimization_tips']:
                print("\nOptimization tips:")
                for tip in recommendations['optimization_tips']:
                    print(f"  • {tip}")
        else:
            print(f"Error: {recommendations['error']}")
        
        # Check if heavy dependencies are available
        if generator.generator is None:
            print("\n" + "="*60)
            print("DEPENDENCY WARNING")
            print("="*60)
            print("Heavy dependencies (langgraph, torch, etc.) are not available.")
            print("Using simple synthetic data generation instead.")
            print("For full functionality, install: pip install -r requirements.txt")
            
            # Load data and use simple generation
            print("\nLoading dataset...")
            data = generator.load_dataset(file_path)
            print("✓ Dataset loaded successfully!")
            
            num_samples = input("Number of synthetic samples to generate (default: 100): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 100
            
            print("Generating synthetic data...")
            result = generator.generate_simple_synthetic_data(data, num_samples)
        else:
            # Run interactive generation with full features
            result = generator.interactive_generation(file_path)
        
        # Display results
        print("\n" + "="*60)
        print("GENERATION RESULTS")
        print("="*60)
        
        if result['success']:
            print("✓ Synthetic data generated successfully!")
            
            if 'synthetic_data' in result:
                synthetic_data = result['synthetic_data']
                if isinstance(synthetic_data, pd.DataFrame):
                    print(f"Generated samples: {len(synthetic_data)}")
                    print(f"Data shape: {synthetic_data.shape}")
                    print(f"Columns: {list(synthetic_data.columns)}")
                    
                    # Save to CSV
                    output_path = "synthetic_data.csv"
                    synthetic_data.to_csv(output_path, index=False)
                    print(f"Output saved to: {output_path}")
                else:
                    print(f"Generated samples: {len(synthetic_data)}")
            
            if 'method' in result:
                print(f"Method used: {result['method']}")
            if 'generation_time' in result:
                print(f"Generation time: {result['generation_time']} seconds")
        else:
            print("✗ Generation failed!")
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 