import os
import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
integration_dir = Path(__file__).parent
src_dir = integration_dir / 'src'
sys.path.append(str(src_dir))

from integration.agent.data_preprocessor import DataPreprocessor
from integration.custom_dataset_generator import CustomDatasetGenerator

# Path to sample data
sample_data_path = integration_dir.parent / 'processed_sample_data.csv'
output_csv_path = integration_dir / 'synthetic_data.csv'

def main():
    print(f"Preprocessing data from: {sample_data_path}")
    preprocessor = DataPreprocessor()
    processed_data, data_type, meta = preprocessor.preprocess_data(sample_data_path)
    print(f"Preprocessing complete. Data type: {data_type}")
    print(f"Meta: {meta}")

    generator = CustomDatasetGenerator()
    requirements = {
        'num_samples': 100,
        'output_format': 'csv',
        'output_path': str(output_csv_path),
    }
    print("Generating synthetic data...")
    result = generator.generate_synthetic_data(processed_data, requirements=requirements)

    if result.get('success'):
        print("Synthetic data generated successfully!")
        synthetic_data = result['synthetic_data']
        if isinstance(synthetic_data, pd.DataFrame):
            synthetic_data.to_csv(output_csv_path, index=False)
            print(f"Synthetic data saved to {output_csv_path}")
        else:
            print("Synthetic data is not a DataFrame. Not saving.")
    else:
        print("Failed to generate synthetic data.")
        print("Error:", result.get('error'))

if __name__ == "__main__":
    main() 