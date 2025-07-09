#!/usr/bin/env python3
"""
Example usage of CustomDatasetGenerator
Demonstrates how to use the flexible interface for different data types and configurations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from integration.custom_dataset_generator import CustomDatasetGenerator

def example_tabular_data():
    """Example with tabular data"""
    print("="*60)
    print("EXAMPLE: TABULAR DATA")
    print("="*60)
    
    # Create sample tabular data
    data = pd.DataFrame({
        'customer_id': range(100),
        'age': np.random.normal(35, 12, 100).astype(int),
        'income': np.random.normal(50000, 20000, 100).astype(int),
        'purchase_amount': np.random.exponential(100, 100).astype(int),
        'category': np.random.choice(['A', 'B', 'C', 'D'], 100, p=[0.4, 0.3, 0.2, 0.1]),
        'satisfaction': np.random.choice(['Low', 'Medium', 'High'], 100, p=[0.2, 0.5, 0.3])
    })
    
    # Save to CSV for demonstration
    data.to_csv('sample_customer_data.csv', index=False)
    
    # Initialize generator
    generator = CustomDatasetGenerator()
    
    # Load the data
    loaded_data = generator.load_dataset('sample_customer_data.csv')
    print(f"Loaded data shape: {loaded_data.shape}")
    
    # Define requirements
    requirements = {
        'domain': 'retail',
        'privacy_level': 'high',  # Will remove customer_id
        'num_samples': 50,
        'gan_type': 'DP-CTGAN',
        'output_format': 'csv',
        'output_path': 'synthetic_customer_data.csv'
    }
    
    # Define metadata
    metadata = {
        'description': 'Customer purchase data for retail analysis',
        'source': 'Internal CRM system',
        'version': '1.0'
    }
    
    # Generate synthetic data
    result = generator.generate_synthetic_data(loaded_data, metadata, requirements)
    
    if result['success']:
        print("✓ Synthetic data generated successfully!")
        print(f"Original columns: {list(data.columns)}")
        print(f"Synthetic columns: {list(result['synthetic_data'].columns)}")
        print(f"Generated samples: {len(result['synthetic_data'])}")
    else:
        print(f"✗ Generation failed: {result['error']}")

def example_image_data():
    """Example with image data"""
    print("\n" + "="*60)
    print("EXAMPLE: IMAGE DATA")
    print("="*60)
    
    # Create sample image data (simulated)
    image_data = []
    for i in range(10):
        # Create random images
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        image_data.append(img)
    
    # Initialize generator
    generator = CustomDatasetGenerator()
    
    # Define requirements for image data
    requirements = {
        'domain': 'research',
        'privacy_level': 'medium',
        'num_samples': 20,
        'gan_type': 'MedGAN',
        'output_format': 'png',
        'output_path': 'synthetic_images'
    }
    
    # Define metadata
    metadata = {
        'description': 'Sample images for computer vision research',
        'source': 'Generated for demonstration',
        'version': '1.0'
    }
    
    # Generate synthetic data
    result = generator.generate_synthetic_data(image_data, metadata, requirements)
    
    if result['success']:
        print("✓ Synthetic images generated successfully!")
        print(f"Original images: {len(image_data)}")
        print(f"Synthetic images: {len(result['synthetic_data'])}")
    else:
        print(f"✗ Generation failed: {result['error']}")

def example_zip_archive():
    """Example with ZIP archive containing multiple files"""
    print("\n" + "="*60)
    print("EXAMPLE: ZIP ARCHIVE")
    print("="*60)
    
    import zipfile
    import tempfile
    
    # Create a ZIP file with multiple data types
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_file:
        zip_path = tmp_file.name
    
    with zipfile.ZipFile(zip_path, 'w') as zip_ref:
        # Add CSV file
        csv_data = pd.DataFrame({
            'id': range(50),
            'value': np.random.normal(100, 20, 50),
            'category': np.random.choice(['X', 'Y', 'Z'], 50)
        })
        csv_data.to_csv('data.csv', index=False)
        zip_ref.write('data.csv')
        
        # Add JSON file
        json_data = {
            'metadata': {
                'description': 'Sample dataset',
                'version': '1.0'
            },
            'data': csv_data.to_dict('records')
        }
        import json
        with open('data.json', 'w') as f:
            json.dump(json_data, f)
        zip_ref.write('data.json')
    
    # Initialize generator
    generator = CustomDatasetGenerator()
    
    try:
        # Load the ZIP archive
        loaded_data = generator.load_dataset(zip_path)
        print(f"Loaded ZIP archive with keys: {list(loaded_data.keys())}")
        
        # Process the first tabular file found
        tabular_files = [k for k, v in loaded_data.items() if isinstance(v, pd.DataFrame)]
        if tabular_files:
            first_file = tabular_files[0]
            data = loaded_data[first_file]
            
            # Define requirements
            requirements = {
                'domain': 'general',
                'privacy_level': 'medium',
                'num_samples': 30,
                'gan_type': 'auto',
                'output_format': 'json',
                'output_path': 'synthetic_from_zip.json'
            }
            
            # Generate synthetic data
            result = generator.generate_synthetic_data(data, {}, requirements)
            
            if result['success']:
                print("✓ Synthetic data generated from ZIP archive!")
                print(f"Generated samples: {len(result['synthetic_data'])}")
            else:
                print(f"✗ Generation failed: {result['error']}")
    
    finally:
        # Cleanup
        import os
        os.unlink(zip_path)
        if os.path.exists('data.csv'):
            os.unlink('data.csv')
        if os.path.exists('data.json'):
            os.unlink('data.json')

def example_programmatic_usage():
    """Example showing programmatic usage without interactive input"""
    print("\n" + "="*60)
    print("EXAMPLE: PROGRAMMATIC USAGE")
    print("="*60)
    
    # Create sample data
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'target': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    })
    
    # Initialize generator
    generator = CustomDatasetGenerator()
    
    # Define different configurations
    configurations = [
        {
            'name': 'High Privacy Healthcare',
            'requirements': {
                'domain': 'healthcare',
                'privacy_level': 'high',
                'num_samples': 100,
                'gan_type': 'DP-CTGAN',
                'output_format': 'csv',
                'output_path': 'healthcare_synthetic.csv'
            },
            'metadata': {
                'description': 'Healthcare data with high privacy',
                'source': 'Medical records',
                'version': '1.0'
            }
        },
        {
            'name': 'Low Privacy Research',
            'requirements': {
                'domain': 'research',
                'privacy_level': 'low',
                'num_samples': 50,
                'gan_type': 'CTGAN',
                'output_format': 'json',
                'output_path': 'research_synthetic.json'
            },
            'metadata': {
                'description': 'Research data with pattern preservation',
                'source': 'Academic study',
                'version': '1.0'
            }
        }
    ]
    
    # Generate synthetic data for each configuration
    for config in configurations:
        print(f"\nGenerating: {config['name']}")
        result = generator.generate_synthetic_data(
            data, 
            config['metadata'], 
            config['requirements']
        )
        
        if result['success']:
            print(f"✓ {config['name']} - Success!")
            print(f"  Samples: {len(result['synthetic_data'])}")
            print(f"  Output: {result['conversion_metadata']['output_path']}")
        else:
            print(f"✗ {config['name']} - Failed: {result['error']}")

def main():
    """Run all examples"""
    print("CUSTOM DATASET GENERATOR - USAGE EXAMPLES")
    print("="*60)
    
    # Run examples
    example_tabular_data()
    example_image_data()
    example_zip_archive()
    example_programmatic_usage()
    
    print("\n" + "="*60)
    print("ALL EXAMPLES COMPLETED")
    print("="*60)
    print("\nTo use with your own data:")
    print("1. Run: python custom_dataset_generator.py")
    print("2. Enter your file path when prompted")
    print("3. Follow the interactive prompts")
    print("\nOr use programmatically:")
    print("from custom_dataset_generator import CustomDatasetGenerator")
    print("generator = CustomDatasetGenerator()")
    print("data = generator.load_dataset('your_file.csv')")
    print("result = generator.generate_synthetic_data(data, metadata, requirements)")

if __name__ == "__main__":
    main() 