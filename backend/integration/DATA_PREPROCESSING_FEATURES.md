# Data Preprocessing Features

This document describes the comprehensive data preprocessing capabilities implemented in the synthetic data platform.

## Overview

The data preprocessing system supports multiple file formats and data types, automatically detecting and processing:
- **ZIP archives** containing images, DICOM files, or tabular data
- **DICOM files** for medical imaging data
- **Individual image files** (JPG, PNG, BMP, TIFF)
- **Tabular data** (CSV, Excel, JSON, Parquet)
- **DataFrames** and lists of data

## Key Features

### 1. ZIP Archive Processing

The system can extract and process ZIP archives containing mixed content:

```python
from agent.data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()

# Process ZIP with tabular focus
processed_data, detected_type, metadata = preprocessor.preprocess_data(
    "data.zip", data_type='tabular'
)

# Process ZIP with image focus
processed_data, detected_type, metadata = preprocessor.preprocess_data(
    "images.zip", data_type='image'
)
```

**Features:**
- Automatic file type detection within archives
- Support for mixed content (images + tabular data)
- Configurable processing limits (max_images, max_dicom_files)
- Comprehensive metadata tracking

### 2. DICOM File Support

Medical imaging data support with pydicom:

```python
# Single DICOM file
processed_data, detected_type, metadata = preprocessor.preprocess_data("scan.dcm")

# DICOM files in ZIP
processed_data, detected_type, metadata = preprocessor.preprocess_data(
    "medical_data.zip", data_type='dicom'
)
```

**Features:**
- Patient metadata extraction
- Pixel data processing
- Study information capture
- Modality detection

### 3. Image Processing

Support for various image formats:

```python
# Single image file
processed_data, detected_type, metadata = preprocessor.preprocess_data("image.jpg")

# List of image arrays
image_arrays = [np.random.randint(0, 255, (64, 64, 3)) for _ in range(10)]
processed_data, detected_type, metadata = preprocessor.preprocess_data(image_arrays)
```

**Supported Formats:**
- JPG/JPEG
- PNG
- BMP
- TIFF/TIF

### 4. Tabular Data Processing

Comprehensive tabular data support:

```python
# CSV files
processed_data, detected_type, metadata = preprocessor.preprocess_data("data.csv")

# Excel files
processed_data, detected_type, metadata = preprocessor.preprocess_data("data.xlsx")

# JSON files
processed_data, detected_type, metadata = preprocessor.preprocess_data("data.json")

# Parquet files
processed_data, detected_type, metadata = preprocessor.preprocess_data("data.parquet")
```

**Features:**
- Automatic format detection
- DataFrame conversion
- Basic data cleaning
- Multiple file combination

### 5. Data Validation

Built-in validation for all supported formats:

```python
# Validate file before processing
validation = preprocessor.validate_file("data.csv")
if validation['valid']:
    processed_data, detected_type, metadata = preprocessor.preprocess_data("data.csv")
```

**Validation Results Include:**
- File existence check
- File size information
- Format support verification
- Error messages for issues

### 6. Error Handling

Comprehensive error handling for various scenarios:

```python
try:
    processed_data, detected_type, metadata = preprocessor.preprocess_data("file.xyz")
except ValueError as e:
    print(f"Unsupported format: {e}")

try:
    processed_data, detected_type, metadata = preprocessor.preprocess_data("nonexistent.csv")
except FileNotFoundError as e:
    print(f"File not found: {e}")
```

## Integration with Synthetic Data Agent

The data preprocessor is fully integrated with the synthetic data generation pipeline:

```python
from agent.synthetic_data_agent import SyntheticDataAgent

agent = SyntheticDataAgent()

# Process data and run pipeline
config = {
    'domain': 'healthcare',
    'privacy_level': 'high',
    'num_samples': 200
}

final_state = agent.preprocess_and_run("medical_data.zip", config)
```

**Integration Features:**
- Automatic data type detection
- Domain-aware processing
- Privacy level consideration
- Quality reporting integration

## Configuration Options

The preprocessor supports various configuration options:

```python
config = {
    'max_images': 1000,        # Limit images for memory management
    'max_dicom_files': 500,    # Limit DICOM files
    'data_type': 'tabular'     # Force specific data type
}

preprocessor = DataPreprocessor(config)
```

## Supported File Formats

### Image Formats
- `.jpg`, `.jpeg`
- `.png`
- `.bmp`
- `.tiff`, `.tif`

### Tabular Formats
- `.csv`
- `.xlsx`, `.xls`
- `.json`
- `.parquet`

### Archive Formats
- `.zip`

### Medical Imaging Formats
- `.dcm`
- `.dicom`

## Metadata Tracking

The system provides comprehensive metadata for all processing operations:

```python
processed_data, detected_type, metadata = preprocessor.preprocess_data("data.zip")

print(f"Source type: {metadata['source_type']}")
print(f"Processing notes: {metadata['processing_notes']}")
print(f"File categories: {metadata['file_categories']}")
print(f"Extracted files: {metadata['extracted_files']}")
```

**Metadata Includes:**
- Source type information
- Processing notes and steps
- File categorization
- Error messages
- Performance metrics

## Usage Examples

### Example 1: Healthcare Data Processing

```python
# Process healthcare data from ZIP
config = {
    'domain': 'healthcare',
    'privacy_level': 'high',
    'data_type': 'tabular'
}

agent = SyntheticDataAgent()
final_state = agent.preprocess_and_run("patient_data.zip", config)

# Generate quality report
quality_report = agent.generate_quality_report(final_state)
print(quality_report)
```

### Example 2: Image Dataset Processing

```python
# Process image dataset
preprocessor = DataPreprocessor({'max_images': 500})

processed_data, detected_type, metadata = preprocessor.preprocess_data(
    "image_dataset.zip", data_type='image'
)

print(f"Processed {len(processed_data)} images")
print(f"Metadata: {metadata['processing_notes']}")
```

### Example 3: Mixed Content Archive

```python
# Process archive with mixed content
preprocessor = DataPreprocessor()

# Extract tabular data
tabular_data, _, tabular_metadata = preprocessor.preprocess_data(
    "mixed_data.zip", data_type='tabular'
)

# Extract images
image_data, _, image_metadata = preprocessor.preprocess_data(
    "mixed_data.zip", data_type='image'
)

print(f"Tabular data shape: {tabular_data.shape}")
print(f"Number of images: {len(image_data)}")
```

## Error Handling Examples

### Unsupported Format
```python
try:
    preprocessor.preprocess_data("file.xyz")
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: Unsupported file format: .xyz
```

### Missing File
```python
try:
    preprocessor.preprocess_data("nonexistent.csv")
except FileNotFoundError as e:
    print(f"Error: {e}")
    # Output: Error: File not found: nonexistent.csv
```

### Empty Data
```python
try:
    preprocessor.preprocess_data([])
except ValueError as e:
    print(f"Error: {e}")
    # Output: Error: Empty data list provided
```

## Dependencies

The system requires the following dependencies:

```txt
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.5.0
seaborn>=0.11.0
Pillow>=9.0.0
pydicom>=2.3.0
openpyxl>=3.0.0
pyarrow>=10.0.0
```

**Optional Dependencies:**
- `pydicom`: Required for DICOM file processing
- `Pillow`: Required for image file processing

## Testing

Run the test suite to verify all features:

```bash
python test_preprocessing_basic.py
```

The test suite demonstrates:
- ZIP archive processing
- Multiple tabular format support
- Image data handling
- Data type detection and validation
- Error handling for unsupported formats
- DICOM file support (when available)
- Comprehensive metadata tracking

## Performance Considerations

- **Memory Management**: Configurable limits for large datasets
- **File Size Limits**: Built-in validation for file sizes
- **Processing Efficiency**: Optimized for common use cases
- **Error Recovery**: Graceful handling of corrupted files

## Future Enhancements

Planned features include:
- Support for additional archive formats (RAR, 7Z)
- Video file processing
- Audio file processing
- Real-time data streaming
- Cloud storage integration
- Parallel processing for large datasets 