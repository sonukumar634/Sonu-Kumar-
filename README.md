from flask import Flask, render_template, request, jsonify, send_file, url_for
import pickle
import numpy as np
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from ai_model.feature_extractor import extract_dockris_features
from utils.docking_utils import run_docking

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'docking'
app.config['PROTEIN_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'proteins')
app.config['LIGAND_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'ligands')
app.config['RESULT_FOLDER'] = os.path.join(app.config['UPLOAD_FOLDER'], 'results')
app.config['ALLOWED_EXTENSIONS'] = {'pdb', 'pdbqt', 'mol2'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Create directories if they don't exist
os.makedirs(app.config['PROTEIN_FOLDER'], exist_ok=True)
os.makedirs(app.config['LIGAND_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load Dockris4 model
with open('ai_model/dockris4_model.pkl', 'rb') as f:
    dockris_model = pickle.load(f)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docking')
def docking():
    return render_template('docking.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'protein' not in request.files or 'ligand' not in request.files:
        return jsonify({'error': 'Missing protein or ligand file'}), 400

    protein_file = request.files['protein']
    ligand_file = request.files['ligand']

    if protein_file.filename == '' or ligand_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if protein_file and allowed_file(protein_file.filename) and \
       ligand_file and allowed_file(ligand_file.filename):

        # Save files with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        protein_filename = f"{timestamp}_protein_{secure_filename(protein_file.filename)}"
        ligand_filename = f"{timestamp}_ligand_{secure_filename(ligand_file.filename)}"

        protein_path = os.path.join(app.config['PROTEIN_FOLDER'], protein_filename)
        ligand_path = os.path.join(app.config['LIGAND_FOLDER'], ligand_filename)

        protein_file.save(protein_path)
        ligand_file.save(ligand_path)

        return jsonify({
            'protein': protein_filename,
            'ligand': ligand_filename,
            'status': 'Files uploaded successfully',
            'timestamp': timestamp
        })

    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or 'protein' not in data or 'ligand' not in data:
        return jsonify({'error': 'Missing protein or ligand filename'}), 400

    protein_path = os.path.join(app.config['PROTEIN_FOLDER'], data['protein'])
    ligand_path = os.path.join(app.config['LIGAND_FOLDER'], data['ligand'])

    # Extract Dockris4 features
    try:
        features = extract_dockris_features(protein_path, ligand_path)
    except Exception as e:
        return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500

    # Make prediction with Dockris4
    try:
        prediction = dockris_model.predict([features])[0]
        prediction = float(prediction)

        # Optional: Run actual docking if needed
        docking_results = None
        if data.get('run_docking', False):
            docking_results = run_docking(protein_path, ligand_path)

        # Create result object
        result = {
            'prediction': prediction,
            'docking_results': docking_results,
            'protein': data['protein'],
            'ligand': data['ligand'],
            'timestamp': data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S")),
            'features': features.tolist()  # Store features for reference
        }

        # Save result to file
        result_filename = f"dockris_result_{result['timestamp']}.json"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)

        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Return result with download link
        return jsonify({
            **result,
            'download_link': url_for('download_result', filename=result_filename)
        })

    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_result(filename):
    try:
        return send_file(
            os.path.join(app.config['RESULT_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 400

@app.route('/results')
def show_results():
    # Get results from query parameters
    prediction = request.args.get('prediction')
    protein = request.args.get('protein')
    ligand = request.args.get('ligand')
    download_link = request.args.get('download_link')

    if not all([prediction, protein, ligand]):
        return redirect(url_for('index'))

    return render_template('result.html',
                         prediction=prediction,
                         protein=protein,
                         ligand=ligand,
                         download_link=download_link)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
