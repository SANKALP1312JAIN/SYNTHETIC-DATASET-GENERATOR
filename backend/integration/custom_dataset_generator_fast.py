#!/usr/bin/env python3
# stylelint-disable
"""
Fast Custom Dataset Synthetic Data Generator
A lightweight interface for generating synthetic data without heavy dependencies
"""

import pandas as pd
import numpy as np
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

class FastCustomDatasetGenerator:
    """Fast interface for custom dataset synthetic data generation without heavy dependencies"""
    
    def __init__(self):
        # Supported file formats
        self.supported_formats = {
            'tabular': ['.csv', '.xlsx', '.json', '.parquet', '.tsv', '.txt'],
            'image': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'],
            'archive': ['.zip', '.tar', '.gz'],
            'dicom': ['.dcm', '.dicom']
        }
    
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
    
    def generate_synthetic_data(self, data: Any, num_samples: int = 100, 
                               method: str = 'statistical') -> Dict[str, Any]:
        """
        Generate synthetic data using simple methods
        
        Args:
            data: Input dataset
            num_samples: Number of synthetic samples to generate
            method: Generation method ('statistical', 'bootstrap', 'smote_simple')
            
        Returns:
            Dictionary containing synthetic data
        """
        try:
            if isinstance(data, pd.DataFrame):
                if method == 'statistical':
                    return self._generate_statistical_synthetic(data, num_samples)
                elif method == 'bootstrap':
                    return self._generate_bootstrap_synthetic(data, num_samples)
                elif method == 'smote_simple':
                    return self._generate_smote_simple_synthetic(data, num_samples)
                else:
                    return self._generate_statistical_synthetic(data, num_samples)
            else:
                return {
                    'success': False,
                    'error': 'Synthetic generation only supports pandas DataFrames'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_statistical_synthetic(self, data: pd.DataFrame, num_samples: int) -> Dict[str, Any]:
        """Generate synthetic data using statistical sampling"""
        synthetic_data = {}
        
        # Process numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            mean_val = data[col].mean()
            std_val = data[col].std()
            if std_val > 0:
                synthetic_data[col] = np.random.normal(mean_val, std_val, num_samples)
            else:
                synthetic_data[col] = np.full(num_samples, mean_val)
        
        # Process categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            categories = data[col].dropna().unique()
            if len(categories) > 0:
                synthetic_data[col] = np.random.choice(categories.tolist(), num_samples)
            else:
                synthetic_data[col] = ['Unknown'] * num_samples
        
        # Process datetime columns
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        for col in datetime_cols:
            min_date = data[col].min()
            max_date = data[col].max()
            date_range = (max_date - min_date).days
            if date_range > 0:
                random_days = np.random.randint(0, date_range, num_samples)
                synthetic_data[col] = [min_date + pd.Timedelta(days=int(days)) for days in random_days]
            else:
                synthetic_data[col] = [min_date] * num_samples
        
        synthetic_df = pd.DataFrame(synthetic_data)
        
        return {
            'success': True,
            'synthetic_data': synthetic_df,
            'method': 'statistical',
            'num_samples': num_samples,
            'original_shape': data.shape,
            'synthetic_shape': synthetic_df.shape
        }
    
    def _generate_bootstrap_synthetic(self, data: pd.DataFrame, num_samples: int) -> Dict[str, Any]:
        """Generate synthetic data using bootstrap sampling"""
        synthetic_data = {}
        
        for col in data.columns:
            # Sample with replacement
            synthetic_data[col] = data[col].sample(n=num_samples, replace=True).values
        
        synthetic_df = pd.DataFrame(synthetic_data)
        
        return {
            'success': True,
            'synthetic_data': synthetic_df,
            'method': 'bootstrap',
            'num_samples': num_samples,
            'original_shape': data.shape,
            'synthetic_shape': synthetic_df.shape
        }
    
    def _generate_smote_simple_synthetic(self, data: pd.DataFrame, num_samples: int) -> Dict[str, Any]:
        """Generate synthetic data using simple SMOTE-like approach"""
        if len(data) < 2:
            return self._generate_statistical_synthetic(data, num_samples)
        
        synthetic_data = {}
        
        # Process numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            values = data[col].dropna().values
            if len(values) >= 2:
                # Simple interpolation between random pairs
                synthetic_values = []
                for _ in range(num_samples):
                    idx1, idx2 = np.random.choice(len(values), 2, replace=False)
                    alpha = np.random.random()
                    synthetic_value = alpha * values[idx1] + (1 - alpha) * values[idx2]
                    synthetic_values.append(synthetic_value)
                synthetic_data[col] = synthetic_values
            else:
                synthetic_data[col] = np.full(num_samples, values[0] if len(values) > 0 else 0)
        
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
            'method': 'smote_simple',
            'num_samples': num_samples,
            'original_shape': data.shape,
            'synthetic_shape': synthetic_df.shape
        }
    
    def save_synthetic_data(self, synthetic_data: pd.DataFrame, output_path: str, 
                           format: str = 'csv') -> Dict[str, Any]:
        """
        Save synthetic data to file
        
        Args:
            synthetic_data: Generated synthetic data
            output_path: Path to save the file
            format: Output format ('csv', 'json', 'xlsx')
            
        Returns:
            Dictionary with save metadata
        """
        try:
            if format == 'csv':
                synthetic_data.to_csv(output_path, index=False)
            elif format == 'json':
                synthetic_data.to_json(output_path, orient='records', indent=2)
            elif format == 'xlsx':
                synthetic_data.to_excel(output_path, index=False)
            else:
                return {
                    'success': False,
                    'error': f'Unsupported format: {format}'
                }
            
            return {
                'success': True,
                'output_path': output_path,
                'format': format,
                'file_size_mb': os.path.getsize(output_path) / (1024 * 1024)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics of the dataset"""
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numerical_summary': {},
            'categorical_summary': {}
        }
        
        # Numerical columns summary
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            summary['numerical_summary'] = data[numerical_cols].describe().to_dict()
        
        # Categorical columns summary
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                summary['categorical_summary'][col] = data[col].value_counts().to_dict()
        
        return summary
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported file formats"""
        return self.supported_formats

def main():
    """Main function for command-line usage"""
    print("FAST CUSTOM DATASET SYNTHETIC DATA GENERATOR")
    print("="*60)
    
    # Initialize generator (should be instant)
    print("Initializing...")
    generator = FastCustomDatasetGenerator()
    print("✓ Initialized successfully!")
    
    # Get file path
    file_path = input("Enter the path to your dataset file: ").strip()
    
    if not file_path:
        print("No file path provided. Exiting.")
        return
    
    try:
        # Load dataset
        print(f"\nLoading dataset from: {file_path}")
        data = generator.load_dataset(file_path)
        print("✓ Dataset loaded successfully!")
        
        # Show data summary
        if isinstance(data, pd.DataFrame):
            summary = generator.get_data_summary(data)
            print(f"Data shape: {summary['shape']}")
            print(f"Columns: {summary['columns']}")
            print(f"Missing values: {sum(summary['missing_values'].values())}")
        
        # Get generation parameters
        num_samples = input(f"Number of synthetic samples to generate (default: 100): ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 100
        
        print("\nAvailable generation methods:")
        methods = ['statistical', 'bootstrap', 'smote_simple']
        for i, method in enumerate(methods, 1):
            print(f"{i}. {method}")
        
        method_choice = input(f"Select method (1-{len(methods)}, default: statistical): ").strip()
        method = methods[int(method_choice) - 1] if method_choice.isdigit() and 1 <= int(method_choice) <= len(methods) else 'statistical'
        
        # Generate synthetic data
        print(f"\nGenerating synthetic data using {method} method...")
        start_time = time.time()
        result = generator.generate_synthetic_data(data, num_samples, method)
        end_time = time.time()
        
        # Display results
        print("\n" + "="*60)
        print("GENERATION RESULTS")
        print("="*60)
        
        if result['success']:
            print("✓ Synthetic data generated successfully!")
            print(f"Method: {result['method']}")
            print(f"Generated samples: {result['num_samples']}")
            print(f"Generation time: {end_time - start_time:.2f} seconds")
            
            synthetic_data = result['synthetic_data']
            print(f"Data shape: {synthetic_data.shape}")
            
            # Save to file
            output_path = f"synthetic_data_{method}.csv"
            save_result = generator.save_synthetic_data(synthetic_data, output_path, 'csv')
            
            if save_result['success']:
                print(f"✓ Output saved to: {output_path}")
                print(f"File size: {save_result['file_size_mb']:.2f} MB")
            else:
                print(f"✗ Save failed: {save_result['error']}")
        else:
            print("✗ Generation failed!")
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 