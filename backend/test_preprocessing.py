#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced data preprocessing capabilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the integration folder to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'integration'))
from ctgan_generator import CTGANSynthesizer

def create_sample_data():
    """Create sample data with categorical columns and missing values"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], n_samples),
        'occupation': np.random.choice(['Engineer', 'Teacher', 'Doctor', 'Sales', 'Manager', 'Designer', 'Analyst', 'Developer'], n_samples),
        'credit_score': np.random.normal(700, 100, n_samples),
        'loan_amount': np.random.normal(250000, 75000, n_samples)
    }
    
    # Add missing values
    data['age'][np.random.choice(n_samples, 50, replace=False)] = np.nan
    data['income'][np.random.choice(n_samples, 30, replace=False)] = np.nan
    data['education'][np.random.choice(n_samples, 20, replace=False)] = np.nan
    data['city'][np.random.choice(n_samples, 15, replace=False)] = np.nan
    
    df = pd.DataFrame(data)
    
    # Save sample data
    sample_file = Path("sample_data.csv")
    df.to_csv(sample_file, index=False)
    
    print(f"Created sample data with {n_samples} samples")
    print(f"Data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Column types:\n{df.dtypes}")
    
    return sample_file

def test_preprocessing():
    """Test the preprocessing pipeline"""
    
    print("=== Testing Enhanced Data Preprocessing ===\n")
    
    # Create sample data
    sample_file = create_sample_data()
    
    # Import the preprocessing function from main.py
    from main import preprocess_data
    
    # Load data
    data = pd.read_csv(sample_file)
    print(f"\nOriginal data shape: {data.shape}")
    
    # Test preprocessing
    try:
        processed_data, preprocessing_info = preprocess_data(data, "test_job")
        
        print(f"\n=== Preprocessing Results ===")
        print(f"Final data shape: {processed_data.shape}")
        print(f"Numerical columns: {preprocessing_info['numerical_columns']}")
        print(f"Categorical columns: {preprocessing_info['categorical_columns']}")
        print(f"Missing values handled: {preprocessing_info['missing_values']}")
        
        print(f"\n=== Preprocessing Steps ===")
        for i, step in enumerate(preprocessing_info['preprocessing_steps'], 1):
            print(f"{i}. {step}")
        
        if preprocessing_info['warnings']:
            print(f"\n=== Warnings ===")
            for warning in preprocessing_info['warnings']:
                print(f"⚠️  {warning}")
        
        # Test CTGAN training
        print(f"\n=== Testing CTGAN Training ===")
        synthesizer = CTGANSynthesizer(epochs=10, batch_size=100)  # Small test
        
        # Get discrete columns (categorical columns after encoding)
        discrete_columns = preprocessing_info.get('categorical_columns', [])
        
        print(f"Training CTGAN with {len(discrete_columns)} discrete columns...")
        processed_file = "processed_sample_data.csv"
        processed_data.to_csv(processed_file, index=False)
        synthesizer.fit(processed_file, discrete_columns=discrete_columns)
        
        print("Generating synthetic data...")
        synthetic_data = synthesizer.generate(100)
        
        print(f"Synthetic data shape: {synthetic_data.shape}")
        print("✅ CTGAN training successful!")
        
        # Clean up
        sample_file.unlink()
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {str(e)}")
        if sample_file.exists():
            sample_file.unlink()

if __name__ == "__main__":
    test_preprocessing() 