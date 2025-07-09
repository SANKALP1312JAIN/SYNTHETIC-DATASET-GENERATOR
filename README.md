# Synthetic Database Generator

A comprehensive web application for generating synthetic data using CTGAN (Conditional Tabular GAN) models. The application features a modern Next.js frontend with a FastAPI backend for machine learning operations and advanced data preprocessing.

## Features

- **Modern UI**: Built with Next.js 15, React 19, and Tailwind CSS
- **CTGAN Integration**: Advanced synthetic data generation using CTGAN models
- **Advanced Data Preprocessing**: Automatic handling of categorical columns, missing values, and data scaling
- **File Upload**: Support for CSV file uploads with intelligent preprocessing
- **Configurable Training**: Adjustable epochs, batch size, and model parameters
- **Real-time Results**: Training progress tracking and detailed preprocessing information
- **Data Export**: Download generated synthetic data in CSV format
- **Template System**: Pre-configured templates for different domains (Healthcare, Finance, Retail)
- **Error Handling**: Comprehensive error reporting and warning system

## Advanced Data Preprocessing

The system automatically handles complex data preprocessing tasks:

### 1. **Missing Value Handling**
- **Numerical columns**: Imputed using mean values
- **Categorical columns**: Imputed using mode (most frequent values)
- **Detailed reporting**: Shows which columns had missing values and how many

### 2. **Categorical Column Processing**
- **Label Encoding**: Converts categorical variables to numerical format
- **High Cardinality Detection**: Identifies columns with too many unique values (>50)
- **Cardinality Reduction**: Reduces high cardinality columns to top 20 categories
- **Preserves Original Values**: Stores encoders for post-processing

### 3. **Numerical Column Scaling**
- **MinMax Scaling**: Scales numerical features to [0,1] range
- **Preserves Original Scale**: Stores scalers for post-processing
- **Maintains Data Distribution**: Ensures synthetic data matches original scale

### 4. **Data Validation**
- **Type Detection**: Automatically identifies numerical vs categorical columns
- **Quality Checks**: Validates data integrity throughout preprocessing
- **Warning System**: Reports potential issues (high cardinality, remaining missing values)

## Project Structure

```
synthetic-db-generator/
├── app/                          # Next.js frontend (App Router)
│   ├── components/               # React components
│   ├── context/                  # React context for state management
│   ├── training-results/         # Training results page
│   └── ...
├── backend/                      # FastAPI backend
│   ├── main.py                   # FastAPI application with preprocessing
│   ├── requirements.txt          # Python dependencies
│   └── test_preprocessing.py     # Test script for preprocessing
├── gans/                         # CTGAN implementation
│   ├── ctgan_generator.py        # CTGAN wrapper class
│   └── ...
└── ...
```

## Prerequisites

- Node.js 18+ and npm/pnpm/yarn
- Python 3.8+ and pip
- Git

## Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd synthetic-db-generator
```

### 2. Frontend Setup

```bash
# Install dependencies
pnpm install  # or npm install / yarn install

# Start development server
pnpm dev      # or npm run dev / yarn dev
```

The frontend will be available at `http://localhost:3000`

### 3. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test preprocessing (optional)
python test_preprocessing.py

# Start FastAPI server
uvicorn main:app --reload --port 8001
```

The backend will be available at `http://localhost:8001`

## Usage

### 1. Upload Data
- Navigate to the home page
- Upload a CSV file using the drag-and-drop interface
- The system will automatically detect and display data summary
- **Supports mixed data types**: numerical and categorical columns

### 2. Configure Model
- Select a domain template (Healthcare, Finance, Retail, or Custom)
- Adjust training parameters:
  - **Epochs**: Number of training iterations (50-500)
  - **Batch Size**: Training batch size (8-128)
  - **Latent Noise**: Noise level for generation (0.01-0.5)
  - **Model Type**: Choose between GAN or VAE

### 3. Train Model
- Click "Train Model" to start the enhanced preprocessing and CTGAN training
- The system will automatically:
  - **Preprocess data**: Handle missing values, encode categories, scale numerical features
  - **Train CTGAN**: Use the preprocessed data with proper discrete column handling
  - **Generate synthetic data**: Create synthetic samples
  - **Post-process**: Reverse scaling and encoding to restore original data format
  - **Redirect to results**: Show detailed preprocessing and training information

### 4. View Results
- **Training Status**: See real-time progress and any errors/warnings
- **Preprocessing Details**: View all preprocessing steps taken
- **Data Information**: Compare original vs synthetic data shapes
- **Download Results**: Get the generated synthetic data in original format

## Data Preprocessing Pipeline

### Automatic Preprocessing Steps

1. **Data Loading**: Read CSV file and validate format
2. **Column Type Detection**: Identify numerical vs categorical columns
3. **Missing Value Analysis**: Count and report missing values
4. **Missing Value Imputation**:
   - Numerical: Mean imputation
   - Categorical: Mode imputation
5. **Categorical Processing**:
   - High cardinality detection (>50 unique values)
   - Cardinality reduction (top 20 categories)
   - Label encoding
6. **Numerical Scaling**: MinMax scaling to [0,1] range
7. **Data Validation**: Final quality checks
8. **CTGAN Training**: Train with proper discrete column specification
9. **Post-processing**: Reverse scaling and encoding

### Supported Data Types

- **Numerical**: int64, float64 (automatically scaled)
- **Categorical**: object, category (automatically encoded)
- **Mixed datasets**: Both numerical and categorical columns
- **Missing values**: Automatically handled
- **High cardinality**: Automatically reduced

## API Integration

### Enhanced FastAPI Backend Endpoints

#### 1. Model Training with Preprocessing
```http
POST /api/train-model
Content-Type: multipart/form-data

Parameters:
- file: CSV file upload (supports mixed data types)
- epochs: int (default: 100)
- batch_size: int (default: 500)
- num_samples: int (default: 1000)

Response includes:
- job_id: Training job identifier
- preprocessing_info: Detailed preprocessing steps
- data_info: Original and synthetic data shapes
- warnings: Any preprocessing warnings
```

#### 2. Job Status with Preprocessing Details
```http
GET /api/job-status/{job_id}

Returns:
- status: "processing", "completed", or "failed"
- progress: Current processing step
- preprocessing_info: Detailed preprocessing information
- errors: Any training errors
- warnings: Preprocessing warnings
```

### Preprocessing Information Structure

```json
{
  "preprocessing_info": {
    "original_shape": [1000, 7],
    "missing_values": {"age": 50, "income": 30},
    "categorical_columns": ["education", "city", "occupation"],
    "numerical_columns": ["age", "income", "credit_score", "loan_amount"],
    "preprocessing_steps": [
      "Column type identification completed",
      "Found missing values in 4 columns",
      "Imputed missing values in numerical columns using mean",
      "Label encoded categorical column 'education'",
      "Scaled numerical columns using MinMaxScaler"
    ],
    "warnings": [
      "High cardinality column 'city' has 10 unique values"
    ],
    "final_shape": [1000, 7]
  }
}
```

## Testing Preprocessing

Run the test script to verify preprocessing capabilities:

```bash
cd backend
python test_preprocessing.py
```

This will:
- Create sample data with missing values and categorical columns
- Test the complete preprocessing pipeline
- Verify CTGAN training with preprocessed data
- Show detailed preprocessing steps and warnings

## Error Handling and Warnings

### Common Warnings
- **High cardinality**: Columns with >50 unique values
- **Missing values**: Columns with null values
- **Data type issues**: Mixed data types in columns

### Error Recovery
- **File format errors**: Invalid CSV files
- **Memory issues**: Large datasets
- **Training failures**: CTGAN convergence issues
- **Post-processing errors**: Scaling/encoding issues

### Debug Information
- **Detailed logs**: Backend console shows preprocessing steps
- **Progress tracking**: Real-time status updates
- **Error details**: Specific error messages and stack traces

## Deployment

### Frontend Deployment (Vercel)

1. **Connect to Vercel:**
   ```bash
   npm install -g vercel
   vercel login
   vercel
   ```

2. **Environment Variables:**
   Set the backend URL in Vercel environment variables:
   ```
   NEXT_PUBLIC_API_URL=https://your-backend-domain.com
   ```

3. **Update API URLs:**
   Replace `http://localhost:8001` with your production backend URL.

### Backend Deployment

#### Option 1: Railway/Heroku
```bash
# Create Procfile
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile

# Deploy
git push heroku main
```

#### Option 2: Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

#### Option 3: AWS/GCP
- Use Cloud Run, App Engine, or EC2
- Set up proper CORS configuration for production
- Use environment variables for configuration

### Production Configuration

1. **CORS Settings:**
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["https://your-frontend-domain.com"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Environment Variables:**
   ```bash
   export API_URL=https://your-backend-domain.com
   export MAX_FILE_SIZE=10485760  # 10MB
   export LOG_LEVEL=INFO
   ```

3. **File Storage:**
   - Use cloud storage (AWS S3, Google Cloud Storage) for file uploads
   - Implement proper cleanup for temporary files
   - Store preprocessing artifacts (scalers, encoders) in persistent storage

## Customization

### Adding New Preprocessing Steps
1. Edit `backend/main.py` preprocessing function
2. Add new preprocessing logic in `preprocess_data()`
3. Update preprocessing information structure
4. Test with `test_preprocessing.py`

### Modifying Preprocessing Parameters
1. **Missing value strategies**: Change imputation methods
2. **Scaling methods**: Use StandardScaler instead of MinMaxScaler
3. **Cardinality thresholds**: Adjust high cardinality detection
4. **Encoding methods**: Use one-hot encoding instead of label encoding

### Styling Changes
- Modify Tailwind classes in components
- Update `tailwind.config.ts` for custom styles
- Edit `app/globals.css` for global styles

## Troubleshooting

### Common Issues

1. **CORS Errors:**
   - Ensure backend CORS settings include your frontend domain
   - Check that the backend is running on the correct port

2. **File Upload Issues:**
   - Verify file format is CSV
   - Check file size limits
   - Ensure file contains valid data

3. **Preprocessing Failures:**
   - Check that uploaded data has valid column types
   - Verify Python dependencies are installed
   - Check backend logs for detailed error messages

4. **Training Failures:**
   - Ensure data has sufficient numerical columns
   - Check for memory issues with large datasets
   - Verify CTGAN parameters are appropriate

5. **Port Conflicts:**
   - Change backend port: `uvicorn main:app --port 8002`
   - Update frontend API calls accordingly

### Debug Mode

Enable debug logging in the backend:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Testing Preprocessing

Use the test script to verify preprocessing:
```bash
cd backend
python test_preprocessing.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CTGAN implementation based on the original paper
- Next.js and React for the frontend framework
- FastAPI for the backend API
- Tailwind CSS for styling
- Scikit-learn for data preprocessing components 