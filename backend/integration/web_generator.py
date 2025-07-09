#!/usr/bin/env python3
"""
Web-based Synthetic Data Generator
Allows drag-and-drop file uploads through a web interface
"""

import csv
import json
import random
import time
import os
from pathlib import Path
from typing import Dict, Any, List
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import tempfile
import zipfile
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()

class WebSyntheticGenerator:
    """Web-based synthetic data generator"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.json', '.txt']
    
    def load_csv_from_bytes(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """Load CSV from bytes"""
        data = []
        try:
            # Try UTF-8 first
            content = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                # Try other encodings
                content = file_bytes.decode('latin-1')
            except:
                content = file_bytes.decode('cp1252')
        
        lines = content.split('\n')
        if not lines:
            return data
        
        # Parse CSV manually
        reader = csv.DictReader(lines)
        for row in reader:
            data.append(row)
        
        return data
    
    def load_json_from_bytes(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """Load JSON from bytes"""
        try:
            content = file_bytes.decode('utf-8')
            data = json.loads(content)
            if isinstance(data, list):
                return data
            else:
                return [data]
        except Exception as e:
            return []
    
    def detect_column_types(self, data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Detect column types"""
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
                'columns': list(column_types.keys()),
                'column_types': column_types
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_csv_bytes(self, data: List[Dict[str, Any]]) -> bytes:
        """Create CSV as bytes"""
        if not data:
            return b''
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue().encode('utf-8')
    
    def create_json_bytes(self, data: List[Dict[str, Any]]) -> bytes:
        """Create JSON as bytes"""
        return json.dumps(data, indent=2).encode('utf-8')

# Create generator instance
generator = WebSyntheticGenerator()

@app.route('/')
def index():
    """Main page"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Synthetic Data Generator</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .upload-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background-color: #fafafa;
                transition: border-color 0.3s;
            }
            .upload-area.dragover {
                border-color: #007bff;
                background-color: #e3f2fd;
            }
            .upload-area input[type="file"] {
                display: none;
            }
            .btn {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            .btn:hover {
                background-color: #0056b3;
            }
            .btn:disabled {
                background-color: #ccc;
                cursor: not-allowed;
            }
            .results {
                margin-top: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
                display: none;
            }
            .progress {
                width: 100%;
                height: 20px;
                background-color: #f0f0f0;
                border-radius: 10px;
                overflow: hidden;
                margin: 10px 0;
            }
            .progress-bar {
                height: 100%;
                background-color: #007bff;
                width: 0%;
                transition: width 0.3s;
            }
            .error {
                color: #dc3545;
                background-color: #f8d7da;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
            .success {
                color: #155724;
                background-color: #d4edda;
                padding: 10px;
                border-radius: 5px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Synthetic Data Generator</h1>
            <p>Upload your CSV or JSON file to generate synthetic data instantly!</p>
            
            <div class="upload-area" id="uploadArea">
                <p>📁 Drag and drop your file here or click to browse</p>
                <input type="file" id="fileInput" accept=".csv,.json,.txt">
                <button class="btn" onclick="document.getElementById('fileInput').click()">Choose File</button>
            </div>
            
            <div id="fileInfo" style="display: none;">
                <h3>File Information</h3>
                <div id="fileDetails"></div>
                
                <div style="margin: 20px 0;">
                    <label for="numSamples">Number of synthetic samples:</label>
                    <input type="number" id="numSamples" value="100" min="1" max="10000" style="margin-left: 10px; padding: 5px;">
                </div>
                
                <button class="btn" id="generateBtn" onclick="generateSyntheticData()">Generate Synthetic Data</button>
            </div>
            
            <div class="progress" id="progress" style="display: none;">
                <div class="progress-bar" id="progressBar"></div>
            </div>
            
            <div class="results" id="results">
                <h3>Results</h3>
                <div id="resultContent"></div>
                <button class="btn" id="downloadBtn" onclick="downloadResults()" style="display: none;">Download Results</button>
            </div>
        </div>

        <script>
            let uploadedFile = null;
            let syntheticData = null;
            
            // Drag and drop functionality
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            });
            
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    handleFile(e.target.files[0]);
                }
            });
            
            function handleFile(file) {
                uploadedFile = file;
                
                // Show file info
                document.getElementById('fileInfo').style.display = 'block';
                document.getElementById('fileDetails').innerHTML = `
                    <p><strong>Name:</strong> ${file.name}</p>
                    <p><strong>Size:</strong> ${(file.size / 1024).toFixed(2)} KB</p>
                    <p><strong>Type:</strong> ${file.type || 'Unknown'}</p>
                `;
            }
            
            function generateSyntheticData() {
                if (!uploadedFile) return;
                
                const numSamples = document.getElementById('numSamples').value;
                const generateBtn = document.getElementById('generateBtn');
                const progress = document.getElementById('progress');
                const progressBar = document.getElementById('progressBar');
                
                generateBtn.disabled = true;
                progress.style.display = 'block';
                
                // Simulate progress
                let progressValue = 0;
                const progressInterval = setInterval(() => {
                    progressValue += 10;
                    progressBar.style.width = progressValue + '%';
                    if (progressValue >= 100) {
                        clearInterval(progressInterval);
                    }
                }, 100);
                
                const formData = new FormData();
                formData.append('file', uploadedFile);
                formData.append('num_samples', numSamples);
                
                fetch('/generate', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    clearInterval(progressInterval);
                    progressBar.style.width = '100%';
                    
                    if (data.success) {
                        syntheticData = data;
                        showResults(data);
                    } else {
                        showError(data.error);
                    }
                    
                    generateBtn.disabled = false;
                    progress.style.display = 'none';
                })
                .catch(error => {
                    clearInterval(progressInterval);
                    showError('Network error: ' + error.message);
                    generateBtn.disabled = false;
                    progress.style.display = 'none';
                });
            }
            
            function showResults(data) {
                const results = document.getElementById('results');
                const resultContent = document.getElementById('resultContent');
                const downloadBtn = document.getElementById('downloadBtn');
                
                resultContent.innerHTML = `
                    <div class="success">
                        <h4>✅ Generation Successful!</h4>
                        <p><strong>Original rows:</strong> ${data.original_count}</p>
                        <p><strong>Synthetic samples:</strong> ${data.num_samples}</p>
                        <p><strong>Columns:</strong> ${data.columns.join(', ')}</p>
                        <p><strong>Generation time:</strong> ${data.generation_time} seconds</p>
                    </div>
                `;
                
                downloadBtn.style.display = 'inline-block';
                results.style.display = 'block';
            }
            
            function showError(message) {
                const results = document.getElementById('results');
                const resultContent = document.getElementById('resultContent');
                
                resultContent.innerHTML = `
                    <div class="error">
                        <h4>❌ Error</h4>
                        <p>${message}</p>
                    </div>
                `;
                
                results.style.display = 'block';
            }
            
            function downloadResults() {
                if (!syntheticData) return;
                
                const format = prompt('Enter format (csv or json):', 'csv').toLowerCase();
                if (format !== 'csv' && format !== 'json') {
                    alert('Invalid format. Please enter "csv" or "json".');
                    return;
                }
                
                const filename = `synthetic_data.${format}`;
                
                fetch('/download', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        data: syntheticData.synthetic_data,
                        format: format
                    })
                })
                .then(response => response.blob())
                .then(blob => {
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                })
                .catch(error => {
                    alert('Download failed: ' + error.message);
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/generate', methods=['POST'])
def generate():
    """Generate synthetic data from uploaded file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        num_samples = int(request.form.get('num_samples', 100))
        
        # Read file content
        file_bytes = file.read()
        file_extension = Path(file.filename).suffix.lower()
        
        # Load data based on file type
        if file_extension == '.csv':
            data = generator.load_csv_from_bytes(file_bytes)
        elif file_extension == '.json':
            data = generator.load_json_from_bytes(file_bytes)
        elif file_extension == '.txt':
            data = generator.load_csv_from_bytes(file_bytes)
        else:
            return jsonify({'success': False, 'error': 'Unsupported file format'})
        
        if not data:
            return jsonify({'success': False, 'error': 'Could not load data from file'})
        
        # Generate synthetic data
        start_time = time.time()
        result = generator.generate_synthetic_data(data, num_samples)
        generation_time = time.time() - start_time
        
        if result['success']:
            result['generation_time'] = round(generation_time, 2)
            return jsonify(result)
        else:
            return jsonify(result)
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download', methods=['POST'])
def download():
    """Download synthetic data"""
    try:
        data = request.json
        synthetic_data = data['data']
        format = data['format']
        
        if format == 'csv':
            file_bytes = generator.create_csv_bytes(synthetic_data)
            return send_file(
                io.BytesIO(file_bytes),
                mimetype='text/csv',
                as_attachment=True,
                download_name='synthetic_data.csv'
            )
        elif format == 'json':
            file_bytes = generator.create_json_bytes(synthetic_data)
            return send_file(
                io.BytesIO(file_bytes),
                mimetype='application/json',
                as_attachment=True,
                download_name='synthetic_data.json'
            )
        else:
            return jsonify({'error': 'Invalid format'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("🌐 Starting Web-based Synthetic Data Generator...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("📁 Drag and drop your CSV or JSON files to generate synthetic data!")
    app.run(debug=True, host='0.0.0.0', port=5000) 