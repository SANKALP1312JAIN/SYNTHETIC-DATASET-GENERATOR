#!/usr/bin/env python3
"""
Minimal Synthetic Data Generator
Ultra-fast startup with only essential dependencies
"""

import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List

class MinimalSyntheticGenerator:
    """Minimal synthetic data generator with instant startup"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.txt']
    
    def select_file(self) -> str:
        """Open file dialog to select a file"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            # Create a root window but hide it
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title="Select your dataset file",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("JSON files", "*.json"),
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
            
            root.destroy()  # Clean up
            return file_path
            
        except ImportError:
            print("tkinter not available, falling back to manual input")
            return input("Enter the path to your CSV/JSON file: ").strip()
        except Exception as e:
            print(f"File dialog error: {e}, falling back to manual input")
            return input("Enter the path to your CSV/JSON file: ").strip()
    
    def load_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Load CSV file without pandas"""
        data = []
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        return data
    
    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        else:
            return [data]
    
    def load_dataset(self, file_path: str) -> List[Dict[str, Any]]:
        """Load dataset from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.csv':
            return self.load_csv(str(file_path))
        elif file_extension == '.json':
            return self.load_json(str(file_path))
        elif file_extension == '.txt':
            return self.load_csv(str(file_path))  # Treat as CSV
        else:
            raise ValueError(f"Unsupported format: {file_extension}")
    
    def detect_column_types(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Detect column types without pandas"""
        if not data:
            return {}
        
        column_types = {}
        sample_row = data[0]
        
        for col, value in sample_row.items():
            if self._is_numeric(value):
                column_types[col] = 'numeric'
            else:
                column_types[col] = 'categorical'
        
        return column_types
    
    def _is_numeric(self, value: Any) -> bool:
        """Check if value is numeric"""
        if value is None or value == '':
            return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    def generate_synthetic_data(self, data: List[Dict[str, Any]], num_samples: int = 100) -> Dict[str, Any]:
        """Generate synthetic data"""
        if not data:
            return {
                'success': False,
                'error': 'No data provided'
            }
        
        try:
            # Detect column types
            column_types = self.detect_column_types(data)
            
            # Get unique values for categorical columns
            categorical_values = {}
            numeric_stats = {}
            
            for col in column_types:
                if column_types[col] == 'categorical':
                    unique_values = list(set(row.get(col, '') for row in data if row.get(col) != ''))
                    categorical_values[col] = unique_values
                else:
                    # Calculate numeric statistics
                    values = [float(row.get(col, 0)) for row in data if self._is_numeric(row.get(col))]
                    if values:
                        numeric_stats[col] = {
                            'mean': sum(values) / len(values),
                            'min': min(values),
                            'max': max(values)
                        }
            
            # Generate synthetic data
            synthetic_data = []
            for _ in range(num_samples):
                synthetic_row = {}
                
                for col in column_types:
                    if column_types[col] == 'categorical':
                        if categorical_values[col]:
                            synthetic_row[col] = random.choice(categorical_values[col])
                        else:
                            synthetic_row[col] = 'Unknown'
                    else:
                        if col in numeric_stats:
                            stats = numeric_stats[col]
                            # Generate random value within range
                            synthetic_row[col] = random.uniform(stats['min'], stats['max'])
                        else:
                            synthetic_row[col] = 0
                
                synthetic_data.append(synthetic_row)
            
            return {
                'success': True,
                'synthetic_data': synthetic_data,
                'num_samples': num_samples,
                'original_count': len(data),
                'columns': list(column_types.keys())
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_to_csv(self, data: List[Dict[str, Any]], output_path: str) -> Dict[str, Any]:
        """Save data to CSV"""
        try:
            if not data:
                return {'success': False, 'error': 'No data to save'}
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
            
            return {
                'success': True,
                'output_path': output_path,
                'rows_saved': len(data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_to_json(self, data: List[Dict[str, Any]], output_path: str) -> Dict[str, Any]:
        """Save data to JSON"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            return {
                'success': True,
                'output_path': output_path,
                'rows_saved': len(data)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def select_save_location(self, default_name: str = "synthetic_data.csv") -> str:
        """Open save dialog to select where to save the output"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            # Create a root window but hide it
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Open save dialog
            file_path = filedialog.asksaveasfilename(
                title="Save synthetic data as",
                defaultextension=".csv",
                filetypes=[
                    ("CSV files", "*.csv"),
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ],
                initialname=default_name
            )
            
            root.destroy()  # Clean up
            return file_path
            
        except ImportError:
            print("tkinter not available, using default location")
            return default_name
        except Exception as e:
            print(f"Save dialog error: {e}, using default location")
            return default_name

def main():
    """Main function"""
    print("MINIMAL SYNTHETIC DATA GENERATOR")
    print("="*50)
    
    # Initialize (should be instant)
    print("Initializing...")
    start_time = time.time()
    generator = MinimalSyntheticGenerator()
    init_time = time.time() - start_time
    print(f"✓ Initialized in {init_time:.3f} seconds")
    
    # Get file path using file dialog
    print("\nSelect your dataset file...")
    file_path = generator.select_file()
    
    if not file_path:
        print("No file selected. Exiting.")
        return
    
    print(f"Selected file: {file_path}")
    
    try:
        # Load data
        print(f"\nLoading data from: {file_path}")
        data = generator.load_dataset(file_path)
        print(f"✓ Loaded {len(data)} rows")
        
        if data:
            print(f"Columns: {list(data[0].keys())}")
        
        # Get number of samples
        num_samples = input(f"Number of synthetic samples to generate (default: 100): ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 100
        
        # Generate synthetic data
        print(f"\nGenerating {num_samples} synthetic samples...")
        start_time = time.time()
        result = generator.generate_synthetic_data(data, num_samples)
        gen_time = time.time() - start_time
        
        # Display results
        print("\n" + "="*50)
        print("RESULTS")
        print("="*50)
        
        if result['success']:
            print("✓ Synthetic data generated successfully!")
            print(f"Generated samples: {result['num_samples']}")
            print(f"Generation time: {gen_time:.2f} seconds")
            print(f"Columns: {result['columns']}")
            
            # Select save location
            print("\nSelect where to save the synthetic data...")
            output_path = generator.select_save_location()
            
            if output_path:
                # Determine format from extension
                if output_path.lower().endswith('.json'):
                    save_result = generator.save_to_json(result['synthetic_data'], output_path)
                else:
                    save_result = generator.save_to_csv(result['synthetic_data'], output_path)
                
                if save_result['success']:
                    print(f"✓ Saved to: {output_path}")
                    print(f"Rows saved: {save_result['rows_saved']}")
                else:
                    print(f"✗ Save failed: {save_result['error']}")
            else:
                print("No save location selected. Data not saved.")
        else:
            print("✗ Generation failed!")
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 