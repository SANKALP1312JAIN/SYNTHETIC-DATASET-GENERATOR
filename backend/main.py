from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import numpy as np
import os
import uuid
import tempfile
from pathlib import Path
import sys
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings
from fastapi.staticfiles import StaticFiles  # <-- Add this line
import statsmodels.api as sm
import shutil
import zipfile
from PIL import Image
warnings.filterwarnings('ignore')

# Add the integration folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'integration'))
from integration.custom_dataset_generator import CustomDatasetGenerator
# Add RBAC and Responsible AI Guardian imports
from integration.agent.rbac_system import rbac_system, Role, Permission
from integration.agent.responsible_ai_guardian import responsible_ai_guardian, PrivacyLevel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for storing uploaded files and generated data
UPLOAD_DIR = Path("uploads")
GENERATED_DIR = Path("generated")
UPLOAD_DIR.mkdir(exist_ok=True)
GENERATED_DIR.mkdir(exist_ok=True)

# Serve the backend directory for static file download
import os
backend_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/files", StaticFiles(directory=backend_dir), name="files")

# Store training jobs
training_jobs = {}

def preprocess_data(data, job_id):
    """
    Advanced data preprocessing for CTGAN training
    """
    preprocessing_info = {
        "original_shape": data.shape,
        "missing_values": {},
        "categorical_columns": [],
        "numerical_columns": [],
        "preprocessing_steps": [],
        "warnings": []
    }
    
    try:
        # Step 1: Identify column types
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        preprocessing_info["numerical_columns"] = numerical_cols
        preprocessing_info["categorical_columns"] = categorical_cols
        preprocessing_info["preprocessing_steps"].append("Column type identification completed")
        
        # Step 2: Handle missing values
        missing_summary = data.isnull().sum()
        columns_with_missing = missing_summary[missing_summary > 0]
        
        if len(columns_with_missing) > 0:
            preprocessing_info["missing_values"] = columns_with_missing.to_dict()
            preprocessing_info["preprocessing_steps"].append(f"Found missing values in {len(columns_with_missing)} columns")
            
            # Handle missing values in numerical columns
            if len(numerical_cols) > 0:
                numerical_data = data[numerical_cols]
                imputer = SimpleImputer(strategy='mean')
                data[numerical_cols] = imputer.fit_transform(numerical_data)
                preprocessing_info["preprocessing_steps"].append("Imputed missing values in numerical columns using mean")
            
            # Handle missing values in categorical columns
            if len(categorical_cols) > 0:
                categorical_data = data[categorical_cols]
                imputer = SimpleImputer(strategy='most_frequent')
                data[categorical_cols] = imputer.fit_transform(categorical_data)
                preprocessing_info["preprocessing_steps"].append("Imputed missing values in categorical columns using mode")
        else:
            preprocessing_info["preprocessing_steps"].append("No missing values found")
        
        # Step 3: Handle categorical columns
        if len(categorical_cols) > 0:
            # Check for high cardinality categorical columns
            high_cardinality_cols = []
            for col in categorical_cols:
                unique_count = data[col].nunique()
                if unique_count > 50:  # Threshold for high cardinality
                    high_cardinality_cols.append(col)
                    preprocessing_info["warnings"].append(f"High cardinality column '{col}' has {unique_count} unique values")
            
            # For high cardinality columns, keep only top categories
            for col in high_cardinality_cols:
                top_categories = data[col].value_counts().head(20).index
                data[col] = data[col].apply(lambda x: x if x in top_categories else 'Other')
                preprocessing_info["preprocessing_steps"].append(f"Reduced cardinality of '{col}' to top 20 categories")
            
            # Encode categorical columns
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
                label_encoders[col] = le
                preprocessing_info["preprocessing_steps"].append(f"Label encoded categorical column '{col}'")
            
            # Save label encoders for later use
            import pickle
            encoders_path = UPLOAD_DIR / f"{job_id}_label_encoders.pkl"
            with open(encoders_path, 'wb') as f:
                pickle.dump(label_encoders, f)
        
        # Step 4: Scale numerical columns
        if len(numerical_cols) > 0:
            scaler = MinMaxScaler()
            data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
            preprocessing_info["preprocessing_steps"].append("Scaled numerical columns using MinMaxScaler")
            
            # Save scaler for later use
            scaler_path = UPLOAD_DIR / f"{job_id}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Step 5: Final validation
        if data.isnull().any().any():
            preprocessing_info["warnings"].append("Some missing values remain after preprocessing")
        
        preprocessing_info["final_shape"] = data.shape
        preprocessing_info["preprocessing_steps"].append("Data preprocessing completed successfully")
        
        return data, preprocessing_info
        
    except Exception as e:
        preprocessing_info["warnings"].append(f"Preprocessing error: {str(e)}")
        raise Exception(f"Data preprocessing failed: {str(e)}")

def is_true_time_series(df, datetime_col='Date'):
    """
    Determines if a DataFrame is true time series or just tabular data with a datetime column.
    Args:
        df (pd.DataFrame): The dataset.
        datetime_col (str): Name of the datetime column (default: 'Date').
    Returns:
        bool: True if true time series, False otherwise.
        str: Reasoning for the decision.
    """
    import pandas as pd
    # Enhancement: Try to find any datetime-like column if the default is missing
    dt_col = datetime_col if datetime_col in df.columns else None
    dt_parse_reason = ""
    if not dt_col:
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                non_null_ratio = parsed.notnull().mean()
                if non_null_ratio > 0.8:
                    dt_col = col
                    dt_parse_reason = f"Column '{col}' detected as datetime (non-null ratio {non_null_ratio:.2f}). "
                    break
            except Exception:
                continue
    if not dt_col:
        return False, "No suitable datetime column found."
    # Try to convert to datetime
    try:
        dt = pd.to_datetime(df[dt_col], errors='coerce')
    except Exception as e:
        return False, f"Failed to parse datetime column: {e}"
    # Remove nulls
    valid = dt.notnull()
    if valid.sum() < 2:
        return False, "Not enough valid datetime values."
    dt = dt[valid]
    # Check if sorted or mostly sorted
    sorted_ratio = (dt.diff().dropna() >= pd.Timedelta(0)).mean()
    if sorted_ratio < 0.8:
        return False, f"Datetime column is not sorted or mostly sorted (sorted_ratio={sorted_ratio:.2f})."
    # High number of unique timestamps
    n_unique = dt.nunique()
    n_total = len(dt)
    if n_unique / n_total < 0.8:
        return False, f"Not enough unique timestamps ({n_unique}/{n_total})."
    # Only one observation per timestamp
    counts = dt.value_counts()
    if (counts > 1).sum() > 0:
        return False, f"Multiple rows per timestamp detected (max count: {counts.max()})."
    # Enhancement: Check if a majority of numeric columns vary over time
    numeric_cols = df.select_dtypes(include=['number']).columns.difference([dt_col])
    if len(numeric_cols) > 0:
        varying_cols = [col for col in numeric_cols if df.loc[valid, col].nunique() > 1]
        if len(varying_cols) < len(numeric_cols) / 2:
            return False, f"Less than half of numeric columns vary over time ({len(varying_cols)}/{len(numeric_cols)})."
        else:
            return True, f"{dt_parse_reason}All time series criteria met. Majority of numeric columns vary over time ({len(varying_cols)}/{len(numeric_cols)})."
    # Heuristic: if datetime is used as measurement axis (not metadata)
    # (Assume if above checks pass, it's likely measurement axis)
    return True, f"{dt_parse_reason}All time series criteria met."


def detect_file_type(file_path, data=None, use_case=None):
    import pandas as pd
    import numpy as np
    from pathlib import Path
    ext = Path(file_path).suffix.lower()
    if ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
        return 'image', 'Detected by file extension.'
    if ext in ['.dcm', '.dicom']:
        return 'dicom', 'Detected by file extension.'
    if ext in ['.zip']:
        return 'multi-table', 'Detected by file extension.'
    if ext in ['.txt']:
        return 'text', 'Detected by file extension.'
    if data is not None:
        if isinstance(data, dict) and all(isinstance(v, pd.DataFrame) for v in data.values()):
            return 'multi-table', 'Detected as multi-table (dict of DataFrames).'
        if isinstance(data, pd.DataFrame):
            is_ts, reason = is_true_time_series(data)
            if is_ts:
                return 'time_series', reason
            return 'tabular', reason
    if ext in ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv', '.txt']:
        return 'tabular', 'Detected by file extension.'
    return 'unknown', 'Could not determine file type.'

def detect_imbalance(data, target_col):
    if target_col and target_col in data.columns:
        counts = data[target_col].value_counts(normalize=True)
        return counts.min() < 0.2  # e.g., minority class < 20%
    return False

def detect_binary_classification(data, target_col):
    if target_col and target_col in data.columns:
        return data[target_col].nunique() == 2
    return False

def find_target_column(data):
    # Heuristic: look for common target column names
    for col in data.columns:
        if col.lower() in ['target', 'label', 'class', 'y', 'outcome']:
            return col
    # Fallback: if only one column is categorical with few unique values
    for col in data.columns:
        if data[col].nunique() <= 10:
            return col
    return None

def select_gan_model(
    data_type,
    industry,
    privacy_level,
    is_imbalanced=False,
    is_medical=False,
    is_multi_table=False,
    is_binary_classification=False,
    use_case=None
):
    # 1. Time Series Rule
    if data_type == 'time_series':
        return 'DoppelGANger'
    if data_type == 'text':
        return 'TextGAN'
    # 2. Privacy Rule
    if privacy_level == 'high' or (use_case and 'privacy' in use_case.lower()):
        return 'DP_CTGAN'
    # 3. Imbalance Rule
    if is_imbalanced or (use_case and 'imbalance' in use_case.lower()):
        return 'SMOTE_CTGAN'
    # 4. Medical/Healthcare
    if is_medical or (industry == 'healthcare'):
        return 'MED_GAN'
    # 5. Multi-table/Relational
    if is_multi_table:
        return 'SDV_GAN'
    # 6. Simple binary classification
    if is_binary_classification:
        return 'Base_GAN'
    # 7. Industry-specific defaults
    if industry == 'retail':
        return 'CTGAN'
    if industry == 'finance':
        return 'CTGAN'
    if industry == 'manufacturing':
        return 'CTGAN'
    # 8. Default for tabular
    if data_type == 'tabular':
        return 'CTGAN'
    # 9. Fallback
    return 'Default_GAN'

@app.get("/api/message/healthcare")
def get_healthcare_message():
    return JSONResponse({"message": "Hello World from Healthcare!"})

@app.get("/api/message/finance")
def get_finance_message():
    return JSONResponse({"message": "Hello World from Finance!"})

@app.get("/api/message/retail")
def get_retail_message():
    return JSONResponse({"message": "Hello World from Retail!"})

@app.get("/api/message/custom")
def get_custom_message():
    return JSONResponse({"message": "Hello World from Custom!"})

# --- AUTHENTICATION ENDPOINTS ---
@app.post("/auth/login")
def login(username: str = Form(...), password: str = Form(...)):
    # For demo, use password as hash (in real use, hash properly)
    session_info = rbac_system.authenticate_user(username, password)
    if session_info:
        return {"success": True, **session_info}
    return JSONResponse({"success": False, "error": "Invalid credentials"}, status_code=401)

@app.post("/auth/logout")
def logout(session_token: str = Form(...)):
    if rbac_system.logout_user(session_token):
        return {"success": True}
    return JSONResponse({"success": False, "error": "Invalid session"}, status_code=401)

@app.post("/auth/register")
def register(username: str = Form(...), email: str = Form(...), role: str = Form(...), password: str = Form(...)):
    try:
        user_id = rbac_system.create_user(username, email, Role(role), password)
        return {"success": True, "user_id": user_id}
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)

# --- RBAC PROTECTED ENDPOINT: GENERATE SYNTHETIC DATA ---
@app.post("/generate-synthetic-data")
async def generate_synthetic_data(
    file: UploadFile = File(...),
    domain: str = Form(...),
    privacy_level: str = Form(...),
    num_samples: int = Form(100),
    output_format: str = Form('csv')
):
    """
    Generate synthetic data using CustomDatasetGenerator and return the file for download.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Save uploaded file
        file_ext = Path(file.filename).suffix
        input_path = os.path.join(temp_dir, f"input{file_ext}")
        with open(input_path, "wb") as f_out:
            f_out.write(await file.read())

        # Load data
        generator = CustomDatasetGenerator()
        data = generator.load_dataset(input_path)

        # Detect file/data type using new logic
        use_case = None  # TODO: set from user input if available
        data_type, reason = detect_file_type(input_path, data, use_case)
        print(f"Detected file/data type: {data_type} ({reason})")

        # --- Enhanced: Detect data metrics for GAN selection ---
        industry = domain.lower() if domain else 'other'
        privacy = privacy_level.lower() if privacy_level else 'medium'
        is_imbalanced = False
        is_binary_classification = False
        is_multi_table = False
        target_col = None
        use_case = None
        is_medical = industry == 'healthcare'
        # Multi-table detection
        if isinstance(data, dict) and all(isinstance(v, pd.DataFrame) for v in data.values()):
            is_multi_table = True
        # Tabular/time series: detect target column, imbalance, binary
        if isinstance(data, pd.DataFrame):
            target_col = find_target_column(data)
            is_imbalanced = detect_imbalance(data, target_col)
            is_binary_classification = detect_binary_classification(data, target_col)
        gan_type = select_gan_model(
            data_type,
            industry,
            privacy,
            is_imbalanced=is_imbalanced,
            is_medical=is_medical,
            is_multi_table=is_multi_table,
            is_binary_classification=is_binary_classification,
            use_case=use_case
        )
        # Override: Use DoppelGANger for time series in finance or healthcare (tracking/sensor)
        if data_type == 'time_series' and (industry == 'finance' or (industry == 'healthcare' and (use_case and any(kw in use_case.lower() for kw in ['sensor', 'device', 'tracking', 'monitor', 'wearable'])))):
            gan_type = 'DoppelGANger'

        backend_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(backend_dir, 'syntheticdata.csv')
        requirements = {
            'domain': domain,
            'privacy_level': privacy_level,
            'num_samples': num_samples,
            'gan_type': gan_type,
            'output_format': output_format,
            'output_path': output_path
        }

        # Generate synthetic data
        result = generator.generate_synthetic_data(data, requirements=requirements)
        print("Result from generator:", result)
        print("Does output_path exist?", os.path.exists(output_path))
        print("output_path:", output_path)

        if not result.get('success'):
            print("Generation failed:", result.get('error'))
            return JSONResponse({
                'success': False,
                'error': result.get('error', 'Unknown error'),
                'details': result
            }, status_code=400)

        # Attach download URL to response
        response = {
            'success': True,
            'download_url': f"/files/syntheticdata.csv",
            'gan_type': gan_type
        }
        return response
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/list-backend-files")
def list_backend_files():
    import os
    return {"files": os.listdir(backend_dir)} 

# Add a helper to make results serializable

def make_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj 