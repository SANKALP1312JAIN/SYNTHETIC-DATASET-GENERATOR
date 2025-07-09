import pandas as pd
import numpy as np
import json
import zipfile
from typing import Dict, Any, List, Tuple, Union, Optional
from pathlib import Path
import warnings
import os
from fastapi.responses import FileResponse
from fastapi import HTTPException
from fastapi.responses import JSONResponse

try:
    import pydicom
    from pydicom.dataset import FileDataset
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("Warning: pydicom not available. DICOM output will not be supported.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Image output will be limited.")

class SyntheticDataGenerator:
    """Generate synthetic data using selected GANs and convert to various formats"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.supported_output_formats = {
            'tabular': ['.csv', '.xlsx', '.json', '.parquet'],
            'image': ['.png', '.jpg', '.jpeg', '.tiff'],
            'dicom': ['.dcm'],
            'archive': ['.zip']
        }
        
    def generate_synthetic_data(self, data: Any, gan_type: str, 
                               num_samples: int, domain: str = 'general',
                               privacy_level: str = 'medium', requirements=None):
        """
        Generate synthetic data using the specified GAN
        
        Args:
            data: Original data for training
            gan_type: Type of GAN to use
            num_samples: Number of synthetic samples to generate
            domain: Domain of the data (healthcare, finance, etc.)
            privacy_level: Privacy level (low, medium, high)
            
        Returns:
            Tuple of (synthetic_data, generation_metadata)
        """
        output_path = requirements.get('output_path', 'synthetic_data.csv') if requirements else 'synthetic_data.csv'
        print(f"Attempting to write synthetic data to: {output_path}")
        generation_metadata = {
            'gan_type': gan_type,
            'num_samples': num_samples,
            'domain': domain,
            'privacy_level': privacy_level,
            'generation_notes': []
        }
        
        try:
            # Determine data type and select appropriate GAN
            data_type = self._detect_data_type(data)
            generation_metadata['data_type'] = data_type
            
            # Apply domain-specific and privacy-specific modifications
            if domain == 'healthcare':
                data = self._apply_healthcare_compliance(data, privacy_level)
                generation_metadata['generation_notes'].append("Applied healthcare compliance measures")
            
            # Generate synthetic data based on type
            if data_type == 'tabular':
                synthetic_data = self._generate_tabular_synthetic(data, gan_type, num_samples, domain, privacy_level)
            elif data_type == 'image':
                synthetic_data = self._generate_image_synthetic(data, gan_type, num_samples, domain, privacy_level)
            elif data_type == 'dicom':
                synthetic_data = self._generate_dicom_synthetic(data, gan_type, num_samples, domain, privacy_level)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            generation_metadata['generation_notes'].append(f"Successfully generated {len(synthetic_data)} synthetic samples")
            generation_metadata['success'] = True
            
            with open(output_path, 'w') as f:
                ...
            print(f"File exists after generation? {os.path.exists(output_path)}")
            
            return synthetic_data, generation_metadata
            
        except Exception as e:
            generation_metadata['generation_notes'].append(f"Error during generation: {str(e)}")
            generation_metadata['success'] = False
            raise
    
    def _detect_data_type(self, data: Any) -> str:
        """Detect the type of data"""
        if isinstance(data, pd.DataFrame):
            return 'tabular'
        elif isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], np.ndarray):
                return 'image'
            elif isinstance(data[0], dict) and 'pixel_data' in data[0]:
                return 'dicom'
            else:
                return 'tabular'
        elif isinstance(data, np.ndarray):
            return 'image'
        else:
            return 'tabular'
    
    def _apply_healthcare_compliance(self, data: Any, privacy_level: str) -> Any:
        """Apply healthcare-specific compliance measures"""
        if privacy_level == 'high':
            # Remove or anonymize sensitive fields
            if isinstance(data, pd.DataFrame):
                sensitive_columns = ['patient_id', 'ssn', 'name', 'address', 'phone']
                for col in sensitive_columns:
                    if col in data.columns:
                        data = data.drop(columns=[col])
                        print(f"Removed sensitive column: {col}")
        
        return data
    
    def _generate_tabular_synthetic(self, data: pd.DataFrame, gan_type: str, 
                                   num_samples: int, domain: str, privacy_level: str) -> pd.DataFrame:
        """Generate synthetic tabular data - OPTIMIZED VERSION"""
        # Vectorized approach for better performance
        synthetic_data = {}
        
        # Process numerical columns in batch
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            numerical_data = data[numerical_cols]
            means = numerical_data.mean()
            stds = numerical_data.std()
            
            # Generate all numerical data at once
            means_array = np.array(means)
            stds_array = np.array(stds)
            synthetic_numerical = np.random.normal(
                means_array.reshape(1, -1), 
                stds_array.reshape(1, -1), 
                (num_samples, len(numerical_cols))
            )
            
            # Apply domain-specific constraints
            if domain == 'healthcare':
                for i, col in enumerate(numerical_cols):
                    if 'age' in col.lower():
                        synthetic_numerical[:, i] = np.clip(synthetic_numerical[:, i], 0, 120)
                    elif 'blood_pressure' in col.lower():
                        synthetic_numerical[:, i] = np.clip(synthetic_numerical[:, i], 60, 200)
                    elif 'cholesterol' in col.lower():
                        synthetic_numerical[:, i] = np.clip(synthetic_numerical[:, i], 100, 400)
            
            for i, col in enumerate(numerical_cols):
                synthetic_data[col] = synthetic_numerical[:, i]
        
        # Process categorical columns in batch
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                categories = data[col].dropna().unique()
                if len(categories) > 0:
                    synthetic_data[col] = np.random.choice(categories.tolist(), num_samples)
                else:
                    synthetic_data[col] = ['Unknown'] * num_samples
        
        return pd.DataFrame(synthetic_data)
    
    def _generate_image_synthetic(self, data: List[np.ndarray], gan_type: str,
                                 num_samples: int, domain: str, privacy_level: str) -> List[np.ndarray]:
        """Generate synthetic image data"""
        synthetic_images = []
        
        # Get reference image shape
        if len(data) > 0:
            ref_shape = data[0].shape
        else:
            ref_shape = (64, 64, 3)  # Default shape
        
        for i in range(num_samples):
            # Generate synthetic image with similar characteristics
            if len(data) > 0:
                # Use statistical properties of original images
                mean_img = np.mean(data, axis=0)
                std_img = np.std(data, axis=0)
                
                # Generate synthetic image with noise
                synthetic_img = np.random.normal(mean_img, std_img)
                synthetic_img = np.clip(synthetic_img, 0, 255).astype(np.uint8)
            else:
                # Generate random image
                synthetic_img = np.random.randint(0, 255, ref_shape, dtype=np.uint8)
            
            synthetic_images.append(synthetic_img)
        
        return synthetic_images
    
    def _generate_dicom_synthetic(self, data: List[Dict], gan_type: str,
                                 num_samples: int, domain: str, privacy_level: str) -> List[Dict]:
        """Generate synthetic DICOM data"""
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM synthetic data generation")
        
        synthetic_dicom_data = []
        
        for i in range(num_samples):
            # Create synthetic DICOM metadata
            synthetic_dicom = {
                'filename': f'synthetic_scan_{i:04d}.dcm',
                'patient_id': f'ANON_{i:06d}',  # Anonymized patient ID
                'study_date': '20240101',  # Generic date
                'modality': 'CT',  # Default modality
                'pixel_data': None,
                'dicom_metadata': {}
            }
            
            # Generate synthetic pixel data if original data available
            if len(data) > 0 and 'pixel_data' in data[0] and data[0]['pixel_data'] is not None:
                ref_pixel_data = data[0]['pixel_data']
                if isinstance(ref_pixel_data, np.ndarray):
                    # Generate synthetic pixel data
                    mean_pixel = np.mean(ref_pixel_data)
                    std_pixel = np.std(ref_pixel_data)
                    synthetic_pixel_data = np.random.normal(mean_pixel, std_pixel, ref_pixel_data.shape)
                    synthetic_pixel_data = np.clip(synthetic_pixel_data, 0, 255).astype(np.uint8)
                    synthetic_dicom['pixel_data'] = synthetic_pixel_data
            
            synthetic_dicom_data.append(synthetic_dicom)
        
        return synthetic_dicom_data
    
    def convert_to_format(self, synthetic_data: Any, output_format: str, 
                         output_path: str, domain: str = 'general',
                         privacy_level: str = 'medium') -> Dict[str, Any]:
        """
        Convert synthetic data to the requested output format
        
        Args:
            synthetic_data: Generated synthetic data
            output_format: Desired output format
            output_path: Path to save the output
            domain: Domain of the data
            privacy_level: Privacy level
            
        Returns:
            Dictionary with conversion metadata
        """
        conversion_metadata = {
            'output_format': output_format,
            'output_path': output_path,
            'domain': domain,
            'privacy_level': privacy_level,
            'conversion_notes': []
        }
        
        try:
            # Apply privacy and domain-specific formatting
            if domain == 'healthcare':
                synthetic_data = self._apply_healthcare_output_compliance(synthetic_data, privacy_level)
                conversion_metadata['conversion_notes'].append("Applied healthcare output compliance")
            
            # Convert based on format
            if output_format.lower() in ['csv', '.csv']:
                self._save_as_csv(synthetic_data, output_path)
            elif output_format.lower() in ['json', '.json']:
                self._save_as_json(synthetic_data, output_path)
            elif output_format.lower() in ['xlsx', 'excel', '.xlsx']:
                self._save_as_excel(synthetic_data, output_path)
            elif output_format.lower() in ['zip', '.zip']:
                self._save_as_zip(synthetic_data, output_path)
            elif output_format.lower() in ['dcm', 'dicom', '.dcm']:
                self._save_as_dicom(synthetic_data, output_path)
            elif output_format.lower() in ['png', 'jpg', 'jpeg', '.png', '.jpg', '.jpeg']:
                self._save_as_images(synthetic_data, output_path, output_format)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
            
            conversion_metadata['conversion_notes'].append(f"Successfully saved to {output_path}")
            conversion_metadata['success'] = True
            
            if not os.path.exists(output_path):
                # Log a clear error
                print(f"File not found: {output_path}")
                return JSONResponse({'success': False, 'error': 'Synthetic data file not found.'}, status_code=500)
            
            return conversion_metadata
            
        except Exception as e:
            conversion_metadata['conversion_notes'].append(f"Error during conversion: {str(e)}")
            conversion_metadata['success'] = False
            raise
    
    def _apply_healthcare_output_compliance(self, data: Any, privacy_level: str) -> Any:
        """Apply healthcare-specific output compliance measures"""
        if privacy_level == 'high':
            # Additional anonymization for output
            if isinstance(data, pd.DataFrame):
                # Remove any remaining sensitive information
                sensitive_patterns = ['id', 'name', 'address', 'phone', 'ssn', 'email']
                for col in data.columns:
                    if any(pattern in col.lower() for pattern in sensitive_patterns):
                        data = data.drop(columns=[col])
                        print(f"Removed sensitive column for output: {col}")
        
        return data
    
    def _save_as_csv(self, data: pd.DataFrame, output_path: str):
        """Save data as CSV"""
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        else:
            # Convert to DataFrame if possible
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
    
    def _save_as_json(self, data: Any, output_path: str):
        """Save data as JSON"""
        if isinstance(data, pd.DataFrame):
            data.to_json(output_path, orient='records', indent=2)
        elif isinstance(data, list):
            # Handle list of dictionaries or arrays
            if len(data) > 0 and isinstance(data[0], dict):
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                # Convert to list of dictionaries
                json_data = [{'data': str(item)} for item in data]
                with open(output_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
        else:
            # Convert to JSON-serializable format
            with open(output_path, 'w') as f:
                json.dump({'data': str(data)}, f, indent=2)
    
    def _save_as_excel(self, data: pd.DataFrame, output_path: str):
        """Save data as Excel"""
        if isinstance(data, pd.DataFrame):
            data.to_excel(output_path, index=False)
        else:
            # Convert to DataFrame if possible
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)
    
    def _save_as_zip(self, data: Any, output_path: str):
        """Save data as ZIP archive"""
        with zipfile.ZipFile(output_path, 'w') as zip_ref:
            if isinstance(data, pd.DataFrame):
                # Save DataFrame as CSV in ZIP
                csv_data = data.to_csv(index=False)
                zip_ref.writestr('synthetic_data.csv', csv_data)
            elif isinstance(data, list):
                # Save multiple files in ZIP
                for i, item in enumerate(data):
                    if isinstance(item, np.ndarray):
                        # Save image as PNG
                        if PIL_AVAILABLE:
                            img = Image.fromarray(item)
                            img_bytes = img.tobytes()
                            zip_ref.writestr(f'image_{i:04d}.png', img_bytes)
                        else:
                            # Save as numpy array
                            np_bytes = item.tobytes()
                            zip_ref.writestr(f'array_{i:04d}.npy', np_bytes)
                    elif isinstance(item, dict):
                        # Save as JSON
                        json_data = json.dumps(item, indent=2, default=str)
                        zip_ref.writestr(f'data_{i:04d}.json', json_data)
            else:
                # Save as generic file
                zip_ref.writestr('synthetic_data.txt', str(data))
    
    def _save_as_dicom(self, data: List[Dict], output_path: str):
        """Save data as DICOM files"""
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM output")
        
        if isinstance(data, list) and len(data) > 0:
            # Create directory for DICOM files
            output_dir = Path(output_path)
            if output_dir.suffix == '.dcm':
                output_dir = output_dir.parent
            output_dir.mkdir(exist_ok=True)
            
            for i, dicom_data in enumerate(data):
                # Create DICOM file
                filename = dicom_data.get('filename', f'synthetic_scan_{i:04d}.dcm')
                filepath = output_dir / filename
                
                # Create basic DICOM dataset
                ds = FileDataset("", {}, file_meta=None, preamble=b"\0" * 128)
                
                # Add basic DICOM attributes
                ds.PatientID = dicom_data.get('patient_id', f'ANON_{i:06d}')
                ds.StudyDate = dicom_data.get('study_date', '20240101')
                ds.Modality = dicom_data.get('modality', 'CT')
                
                # Add pixel data if available
                if 'pixel_data' in dicom_data and dicom_data['pixel_data'] is not None:
                    pixel_data = dicom_data['pixel_data']
                    ds.Rows = pixel_data.shape[0]
                    ds.Columns = pixel_data.shape[1]
                    ds.BitsAllocated = 8
                    ds.BitsStored = 8
                    ds.HighBit = 7
                    ds.PixelRepresentation = 0
                    ds.SamplesPerPixel = 1 if len(pixel_data.shape) == 2 else pixel_data.shape[2]
                    ds.PhotometricInterpretation = "MONOCHROME2" if len(pixel_data.shape) == 2 else "RGB"
                    ds.PixelData = pixel_data.tobytes()
                
                ds.save_as(filepath)
        else:
            raise ValueError("DICOM output requires list of DICOM data dictionaries")
    
    def _save_as_images(self, data: List[np.ndarray], output_path: str, format_ext: str):
        """Save data as image files"""
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow required for image output")
        
        if isinstance(data, list) and len(data) > 0:
            # Create directory for images
            output_dir = Path(output_path)
            if output_dir.suffix in ['.png', '.jpg', '.jpeg']:
                output_dir = output_dir.parent
            output_dir.mkdir(exist_ok=True)
            
            # Determine file extension
            if format_ext.startswith('.'):
                ext = format_ext
            else:
                ext = f'.{format_ext}'
            
            for i, img_array in enumerate(data):
                if isinstance(img_array, np.ndarray):
                    img = Image.fromarray(img_array)
                    filename = f'synthetic_image_{i:04d}{ext}'
                    filepath = output_dir / filename
                    img.save(filepath)
        else:
            raise ValueError("Image output requires list of numpy arrays")
    
    def get_supported_output_formats(self) -> Dict[str, List[str]]:
        """Get list of supported output formats"""
        return self.supported_output_formats
    
    def validate_output_format(self, format_name: str, data_type: str) -> bool:
        """Validate if output format is supported for data type"""
        if data_type in self.supported_output_formats:
            return format_name.lower() in [f.lower() for f in self.supported_output_formats[data_type]]
        return False
