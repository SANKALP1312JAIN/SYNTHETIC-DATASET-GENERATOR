import pandas as pd
import numpy as np
import zipfile
import json
import os
from typing import Dict, Any, List, Tuple, Union, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("Warning: pydicom not available. DICOM files will not be supported.")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL/Pillow not available. Image processing will be limited.")

class DataPreprocessor:
    """Comprehensive data preprocessor for various file formats"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        self.supported_tabular_formats = ['.csv', '.xlsx', '.xls', '.json', '.parquet']
        self.supported_archive_formats = ['.zip']
        self.supported_dicom_formats = ['.dcm', '.dicom']
        
    def preprocess_data(self, data_source: Union[str, Path, pd.DataFrame, List], 
                       data_type: Optional[str] = None) -> Tuple[Any, str, Dict]:
        """
        Main preprocessing function that handles various data sources
        
        Args:
            data_source: File path, DataFrame, or list of data
            data_type: Optional hint for data type ('tabular', 'image', 'dicom')
            
        Returns:
            Tuple of (processed_data, detected_type, metadata)
        """
        if isinstance(data_source, (str, Path)):
            return self._preprocess_file(data_source, data_type)
        elif isinstance(data_source, pd.DataFrame):
            return self._preprocess_dataframe(data_source)
        elif isinstance(data_source, list):
            return self._preprocess_list(data_source, data_type)
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
    
    def _preprocess_file(self, file_path: Union[str, Path], data_type: Optional[str] = None) -> Tuple[Any, str, Dict]:
        """Preprocess data from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        # Handle archive files (ZIP)
        if file_extension in self.supported_archive_formats:
            return self._preprocess_archive(file_path, data_type)
        
        # Handle DICOM files
        elif file_extension in self.supported_dicom_formats:
            return self._preprocess_dicom(file_path)
        
        # Handle image files
        elif file_extension in self.supported_image_formats:
            return self._preprocess_image_file(file_path)
        
        # Handle tabular files
        elif file_extension in self.supported_tabular_formats:
            return self._preprocess_tabular_file(file_path)
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _preprocess_archive(self, archive_path: Path, data_type: Optional[str] = None) -> Tuple[Any, str, Dict]:
        """Preprocess ZIP archive containing images or data files"""
        metadata = {
            'source_type': 'archive',
            'file_path': str(archive_path),
            'processing_notes': []
        }
        
        try:
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Extract images
                image_files = [f for f in zip_ref.namelist() 
                             if Path(f).suffix.lower() in self.supported_image_formats]
                if image_files:
                    images = self._extract_images_from_archive(zip_ref, image_files)
                    metadata['processing_notes'].append(f"Extracted {len(images)} images")
                    return images, 'image', metadata
                
                # Extract DICOM files
                dicom_files = [f for f in zip_ref.namelist() 
                             if Path(f).suffix.lower() in self.supported_dicom_formats]
                if dicom_files:
                    dicom_data = self._extract_dicom_from_archive(zip_ref, dicom_files)
                    metadata['processing_notes'].append(f"Extracted {len(dicom_data)} DICOM files")
                    return dicom_data, 'dicom', metadata
                
                # Extract tabular files
                tabular_files = [f for f in zip_ref.namelist() 
                               if Path(f).suffix.lower() in self.supported_tabular_formats]
                if tabular_files:
                    dfs = []
                    for file_name in tabular_files:
                        with zip_ref.open(file_name) as f:
                            df = self._read_tabular_file(f, Path(file_name).suffix)
                            dfs.append(df)
                    
                    if len(dfs) == 1:
                        metadata['processing_notes'].append("Extracted single tabular file")
                        return dfs[0], 'tabular', metadata
                    else:
                        combined_df = pd.concat(dfs, ignore_index=True)
                        metadata['processing_notes'].append(f"Combined {len(dfs)} tabular files")
                        return combined_df, 'tabular', metadata
                
                raise ValueError("No supported files found in archive")
                
        except Exception as e:
            metadata['processing_notes'].append(f"Error processing archive: {str(e)}")
            raise
    
    def _extract_images_from_archive(self, zip_ref: zipfile.ZipFile, image_files: List[str]) -> List[np.ndarray]:
        """Extract images from ZIP archive"""
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow required for image processing")
        
        images = []
        max_images = self.config.get('max_images', 1000)  # Limit for memory
        
        for i, image_file in enumerate(image_files[:max_images]):
            try:
                with zip_ref.open(image_file) as file:
                    img = Image.open(file)
                    img_array = np.array(img)
                    images.append(img_array)
            except Exception as e:
                print(f"Warning: Could not process image {image_file}: {e}")
                continue
        
        return images
    
    def _extract_dicom_from_archive(self, zip_ref: zipfile.ZipFile, dicom_files: List[str]) -> List[Dict]:
        """Extract DICOM files from ZIP archive"""
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM processing")
        
        dicom_data = []
        max_files = self.config.get('max_dicom_files', 500)
        
        for i, dicom_file in enumerate(dicom_files[:max_files]):
            try:
                with zip_ref.open(dicom_file) as file:
                    dcm = pydicom.dcmread(file)
                    # Extract pixel data and metadata
                    pixel_data = dcm.pixel_array if hasattr(dcm, 'pixel_array') else None
                    metadata = {
                        'filename': dicom_file,
                        'patient_id': getattr(dcm, 'PatientID', 'Unknown'),
                        'study_date': getattr(dcm, 'StudyDate', 'Unknown'),
                        'modality': getattr(dcm, 'Modality', 'Unknown'),
                        'pixel_data': pixel_data,
                        'dicom_metadata': dict(dcm)
                    }
                    dicom_data.append(metadata)
            except Exception as e:
                print(f"Warning: Could not process DICOM file {dicom_file}: {e}")
                continue
        
        return dicom_data
    
    def _preprocess_dicom(self, dicom_path: Path) -> Tuple[List[Dict], str, Dict]:
        """Preprocess single DICOM file"""
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM processing")
        
        metadata = {
            'source_type': 'dicom',
            'file_path': str(dicom_path),
            'processing_notes': []
        }
        
        try:
            dcm = pydicom.dcmread(dicom_path)
            pixel_data = dcm.pixel_array if hasattr(dcm, 'pixel_array') else None
            
            dicom_data = [{
                'filename': dicom_path.name,
                'patient_id': getattr(dcm, 'PatientID', 'Unknown'),
                'study_date': getattr(dcm, 'StudyDate', 'Unknown'),
                'modality': getattr(dcm, 'Modality', 'Unknown'),
                'pixel_data': pixel_data,
                'dicom_metadata': dict(dcm)
            }]
            
            metadata['processing_notes'].append("Successfully processed DICOM file")
            return dicom_data, 'dicom', metadata
            
        except Exception as e:
            metadata['processing_notes'].append(f"Error processing DICOM: {str(e)}")
            raise
    
    def _preprocess_image_file(self, image_path: Path) -> Tuple[List[np.ndarray], str, Dict]:
        """Preprocess single image file"""
        if not PIL_AVAILABLE:
            raise ImportError("PIL/Pillow required for image processing")
        
        metadata = {
            'source_type': 'image',
            'file_path': str(image_path),
            'processing_notes': []
        }
        
        try:
            img = Image.open(image_path)
            img_array = np.array(img)
            images = [img_array]
            
            metadata['processing_notes'].append("Successfully processed image file")
            return images, 'image', metadata
            
        except Exception as e:
            metadata['processing_notes'].append(f"Error processing image: {str(e)}")
            raise
    
    def _preprocess_tabular_file(self, file_path: Path) -> Tuple[pd.DataFrame, str, Dict]:
        """Preprocess tabular file (CSV, Excel, JSON)"""
        metadata = {
            'source_type': 'tabular',
            'file_path': str(file_path),
            'processing_notes': []
        }
        
        try:
            df = self._read_tabular_file(file_path, file_path.suffix)
            metadata['processing_notes'].append("Successfully processed tabular file")
            return df, 'tabular', metadata
            
        except Exception as e:
            metadata['processing_notes'].append(f"Error processing tabular file: {str(e)}")
            raise
    
    def _read_tabular_file(self, file_path: Union[Path, Any], file_extension: str) -> pd.DataFrame:
        """Read tabular file based on extension"""
        file_extension = file_extension.lower()
        
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_extension == '.json':
            return pd.read_json(file_path)
        elif file_extension == '.parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported tabular format: {file_extension}")
    
    def _preprocess_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str, Dict]:
        """Preprocess existing DataFrame"""
        metadata = {
            'source_type': 'dataframe',
            'processing_notes': ["DataFrame provided directly"]
        }
        
        # Basic DataFrame validation and cleaning
        df_cleaned = self._clean_dataframe(df)
        metadata['processing_notes'].append(f"Cleaned DataFrame: {df.shape} -> {df_cleaned.shape}")
        
        return df_cleaned, 'tabular', metadata
    
    def _preprocess_list(self, data_list: List, data_type: Optional[str] = None) -> Tuple[Any, str, Dict]:
        """Preprocess list of data"""
        metadata = {
            'source_type': 'list',
            'processing_notes': []
        }
        
        if not data_list:
            raise ValueError("Empty data list provided")
        
        # Try to determine type from first element
        first_element = data_list[0]
        
        if isinstance(first_element, np.ndarray) or data_type == 'image':
            # Image data
            metadata['processing_notes'].append(f"Processing {len(data_list)} images")
            return data_list, 'image', metadata
        
        elif isinstance(first_element, dict) or data_type == 'dicom':
            # DICOM or structured data
            metadata['processing_notes'].append(f"Processing {len(data_list)} structured records")
            return data_list, 'dicom', metadata
        
        elif isinstance(first_element, (list, tuple)) and len(first_element) > 1:
            # Convert to DataFrame if possible
            try:
                df = pd.DataFrame(data_list)
                metadata['processing_notes'].append(f"Converted list to DataFrame: {df.shape}")
                return df, 'tabular', metadata
            except:
                metadata['processing_notes'].append("Could not convert to DataFrame, treating as raw list")
                return data_list, 'unknown', metadata
        
        else:
            # Single column data
            df = pd.DataFrame(data_list)
            metadata['processing_notes'].append(f"Converted to single-column DataFrame: {df.shape}")
            return df, 'tabular', metadata
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic DataFrame cleaning"""
        df_cleaned = df.copy()
        
        # Remove completely empty rows and columns
        df_cleaned = df_cleaned.dropna(how='all').dropna(axis=1, how='all')
        
        # Reset index
        df_cleaned = df_cleaned.reset_index(drop=True)
        
        return df_cleaned
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get list of supported file formats"""
        return {
            'image': self.supported_image_formats,
            'tabular': self.supported_tabular_formats,
            'archive': self.supported_archive_formats,
            'dicom': self.supported_dicom_formats
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate if file can be processed"""
        file_path = Path(file_path)
        
        validation_result = {
            'valid': False,
            'file_path': str(file_path),
            'file_exists': file_path.exists(),
            'file_size': None,
            'supported_format': False,
            'format_type': None,
            'error_message': None
        }
        
        if not validation_result['file_exists']:
            validation_result['error_message'] = "File does not exist"
            return validation_result
        
        try:
            validation_result['file_size'] = file_path.stat().st_size
            file_extension = file_path.suffix.lower()
            
            if file_extension in self.supported_image_formats:
                validation_result['supported_format'] = True
                validation_result['format_type'] = 'image'
            elif file_extension in self.supported_tabular_formats:
                validation_result['supported_format'] = True
                validation_result['format_type'] = 'tabular'
            elif file_extension in self.supported_archive_formats:
                validation_result['supported_format'] = True
                validation_result['format_type'] = 'archive'
            elif file_extension in self.supported_dicom_formats:
                validation_result['supported_format'] = True
                validation_result['format_type'] = 'dicom'
            else:
                validation_result['error_message'] = f"Unsupported file format: {file_extension}"
                return validation_result
            
            validation_result['valid'] = True
            
        except Exception as e:
            validation_result['error_message'] = str(e)
        
        return validation_result
