# Synthetic Data Generation Platform

A comprehensive, privacy-preserving synthetic data generation platform that automatically selects and implements domain-aware GANs (Generative Adversarial Networks) to create high-quality synthetic datasets while maintaining data characteristics and ensuring privacy compliance.

## 🚀 Quick Start

### Interactive Mode (Recommended for Beginners)
```bash
python custom_dataset_generator.py
```
Simply run this command and follow the prompts to upload your dataset and generate synthetic data!

### Programmatic Mode (For Developers)
```python
from custom_dataset_generator import CustomDatasetGenerator

# Load your dataset
generator = CustomDatasetGenerator()
data = generator.load_dataset("your_data.csv")

# Generate synthetic data
result = generator.generate_synthetic_data(
    data=data,
    requirements={
        'domain': 'healthcare',
        'privacy_level': 'high',
        'num_samples': 100,
        'gan_type': 'DP-CTGAN',
        'output_format': 'csv',
        'output_path': 'synthetic_data.csv'
    }
)
```

## ✨ Key Features

- **🔄 Automatic GAN Selection**: Intelligently chooses the best GAN based on your data type and domain
- **🔒 Privacy-Preserving**: Built-in differential privacy and anonymization for sensitive data
- **📊 Multi-Format Support**: Handles CSV, Excel, JSON, images, DICOM, and ZIP archives
- **🎯 Domain-Aware**: Specialized handling for healthcare, finance, retail, and other domains
- **📈 Quality Assessment**: Automatic quality evaluation and reporting
- **🛡️ Compliance Ready**: HIPAA, GDPR, and other regulatory compliance features
- **📋 Comprehensive Reporting**: Detailed quality and privacy reports

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies
```bash
# Clone the repository
git clone <repository-url>
cd synthetic_data_platform

# Install required packages
pip install pandas numpy scikit-learn torch torchvision matplotlib seaborn

# Optional: For advanced features (DICOM, Excel support)
pip install pydicom pillow openpyxl
```

## 📁 Supported File Formats

### Tabular Data
- **CSV** (`.csv`) - Comma-separated values
- **Excel** (`.xlsx`, `.xls`) - Microsoft Excel files
- **JSON** (`.json`) - JavaScript Object Notation
- **Parquet** (`.parquet`) - Columnar storage format
- **TSV** (`.tsv`, `.txt`) - Tab-separated values

### Image Data
- **PNG** (`.png`) - Portable Network Graphics
- **JPEG** (`.jpg`, `.jpeg`) - Joint Photographic Experts Group
- **TIFF** (`.tiff`, `.tif`) - Tagged Image File Format
- **BMP** (`.bmp`) - Bitmap
- **GIF** (`.gif`) - Graphics Interchange Format

### Archives
- **ZIP** (`.zip`) - Compressed archives containing multiple files

### Medical Data
- **DICOM** (`.dcm`, `.dicom`) - Medical imaging format (requires pydicom)

## 🎯 Usage Examples

### Example 1: Healthcare Data
```python
import pandas as pd
import numpy as np

# Create sample healthcare data
healthcare_data = pd.DataFrame({
    'patient_id': range(100),
    'age': np.random.normal(45, 15, 100).astype(int),
    'blood_pressure': np.random.normal(120, 20, 100).astype(int),
    'cholesterol': np.random.normal(200, 40, 100).astype(int),
    'diagnosis': np.random.choice(['A', 'B', 'C', 'D'], 100, p=[0.6, 0.2, 0.15, 0.05]),
    'treatment': np.random.choice(['Medication', 'Surgery', 'Therapy'], 100)
})

healthcare_data.to_csv('healthcare_data.csv', index=False)

# Generate synthetic data with high privacy
from custom_dataset_generator import CustomDatasetGenerator

generator = CustomDatasetGenerator()
data = generator.load_dataset('healthcare_data.csv')

result = generator.generate_synthetic_data(
    data=data,
    requirements={
        'domain': 'healthcare',
        'privacy_level': 'high',  # Will remove patient_id
        'num_samples': 50,
        'gan_type': 'DP-CTGAN',
        'output_format': 'csv',
        'output_path': 'synthetic_healthcare.csv'
    }
)

print(f"Generated {len(result['synthetic_data'])} synthetic samples")
```

### Example 2: Financial Data
```python
# Create sample financial data
financial_data = pd.DataFrame({
    'customer_id': range(200),
    'income': np.random.normal(75000, 25000, 200),
    'credit_score': np.random.normal(700, 100, 200).astype(int),
    'loan_amount': np.random.exponential(50000, 200),
    'risk_category': np.random.choice(['Low', 'Medium', 'High'], 200, p=[0.6, 0.3, 0.1])
})

financial_data.to_csv('financial_data.csv', index=False)

# Generate synthetic data
result = generator.generate_synthetic_data(
    data=financial_data,
    requirements={
        'domain': 'finance',
        'privacy_level': 'medium',
        'num_samples': 100,
        'gan_type': 'CTGAN',
        'output_format': 'json',
        'output_path': 'synthetic_financial.json'
    }
)
```

### Example 3: Image Data
```python
import numpy as np
from PIL import Image
import zipfile

# Create sample images and save as ZIP
image_data = []
for i in range(10):
    img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    image_data.append(img_array)

# Save as ZIP archive
with zipfile.ZipFile('sample_images.zip', 'w') as zip_ref:
    for i, img in enumerate(image_data):
        img_pil = Image.fromarray(img)
        img_bytes = img_pil.tobytes()
        zip_ref.writestr(f'image_{i}.png', img_bytes)

# Generate synthetic images
result = generator.generate_synthetic_data(
    data=image_data,
    requirements={
        'domain': 'research',
        'privacy_level': 'medium',
        'num_samples': 20,
        'gan_type': 'MedGAN',
        'output_format': 'png',
        'output_path': 'synthetic_images'
    }
)
```

## 🔧 Configuration Options

### Domain Settings
| Domain | Privacy Level | Recommended GAN | Use Case |
|--------|---------------|-----------------|----------|
| `healthcare` | High | DP-CTGAN | Medical records, patient data |
| `finance` | Medium | CTGAN | Banking, credit data |
| `retail` | Medium | CTGAN | Customer transactions |
| `education` | High | DP-CTGAN | Student records |
| `research` | Low | SDVGAN | Academic datasets |
| `general` | Medium | DefaultGAN | Generic data |

### Privacy Levels
- **`low`**: Minimal privacy protection, preserves original patterns
- **`medium`**: Balanced privacy and utility
- **`high`**: Maximum privacy, removes sensitive information

### GAN Types
- **`auto`**: Automatic selection based on data type and domain
- **`DP-CTGAN`**: Differentially private GAN for sensitive data
- **`CTGAN`**: Standard conditional tabular GAN
- **`MedGAN`**: Medical data specific GAN
- **`DefaultGAN`**: Simple fallback for unknown data types

### Output Formats
- **`csv`**: Comma-separated values
- **`json`**: JavaScript Object Notation
- **`xlsx`**: Excel format
- **`zip`**: Compressed archive
- **`png`**: Image format (for image data)
- **`dcm`**: DICOM format (for medical data)

## 🏗️ Architecture

The platform uses a modular, agent-based architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  Data Profiler  │───▶│  GAN Selector   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Quality Report  │◀───│ Quality Eval.   │◀───│ Synthetic Data  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Privacy Report  │    │ Mode Collapse   │    │ Output Formats  │
└─────────────────┘    │   Detection     │    └─────────────────┘
                       └─────────────────┘
```

### Core Components

1. **Data Preprocessor**: Handles file loading and format conversion
2. **Data Profiler**: Analyzes data characteristics and quality
3. **GAN Selector**: Chooses appropriate GAN based on domain and data type
4. **Synthetic Data Generator**: Creates synthetic samples using selected GAN
5. **Quality Evaluator**: Assesses synthetic data quality and similarity
6. **Privacy Guardian**: Ensures privacy compliance and protection
7. **Mode Collapse Detector**: Monitors GAN training for quality issues

## 📊 Quality Assessment

The platform automatically evaluates synthetic data quality using:

- **Statistical Similarity**: Distribution comparison between original and synthetic data
- **Privacy Metrics**: Differential privacy and anonymization assessment
- **Data Utility**: Preservation of important data characteristics
- **Mode Collapse Detection**: Identifies GAN training issues
- **Domain-Specific Metrics**: Healthcare, finance, and other domain-specific evaluations

## 🔒 Privacy & Compliance

### Built-in Privacy Features
- **Differential Privacy**: Configurable epsilon and delta parameters
- **Data Anonymization**: Automatic removal of sensitive identifiers
- **Noise Injection**: Adds controlled noise for privacy protection
- **Access Control**: Role-based access control system

### Compliance Frameworks
- **HIPAA**: Healthcare data privacy compliance
- **GDPR**: European data protection regulation
- **SOX**: Financial data compliance
- **FERPA**: Educational data privacy

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test synthetic data generation
python test_synthetic_generation_simple.py

# Test data preprocessing
python test_data_preprocessing.py

# Test with your own data
python example_usage.py
```

## 📈 Performance

### Typical Performance Metrics
- **Small datasets** (< 1K rows): 10-30 seconds
- **Medium datasets** (1K-10K rows): 1-5 minutes
- **Large datasets** (> 10K rows): 5-15 minutes

### Memory Requirements
- **Minimum**: 4GB RAM
- **Recommended**: 8GB+ RAM for large datasets
- **GPU**: Optional, accelerates training with CUDA support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

**Q: "Module not found" errors**
A: Install missing dependencies: `pip install -r requirements.txt`

**Q: DICOM files not supported**
A: Install pydicom: `pip install pydicom`

**Q: Excel files not loading**
A: Install openpyxl: `pip install openpyxl`

**Q: Memory errors with large datasets**
A: Reduce batch size or use smaller sample size

### Getting Help
- Check the test files for usage examples
- Review the `example_usage.py` file
- Run the interactive mode for guided usage

---

**Ready to generate synthetic data?** Start with `python custom_dataset_generator.py` and follow the prompts! 
