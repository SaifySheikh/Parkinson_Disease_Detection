from flask import Blueprint, render_template, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
from app.voice_model import voice_model_feature_extraction  # Import your function
from app.image_model import image_model_feature_extraction  # Import your function

main_routes = Blueprint('main_routes', __name__)

@main_routes.route('/')
def home():
    return render_template('index.html')

@main_routes.route('/index')
def home1():
    return render_template('index.html')

@main_routes.route('/developer')
def developer():
    return render_template('developer.html')

@main_routes.route('/upload', methods=['POST'])
def upload():
    # Ensure both files (voice and image) are uploaded
    if 'voiceFile' not in request.files or 'imageFile' not in request.files:
        return jsonify({'error': 'Both voice and image files are required'}), 400

    # Get the uploaded files
    voice_file = request.files['voiceFile']
    image_file = request.files['imageFile']

    # Check if both files were selected
    if voice_file.filename == '' or image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Secure the filenames to prevent directory traversal attacks
    voice_filename = secure_filename(voice_file.filename)
    image_filename = secure_filename(image_file.filename)

    # Define allowed extensions
    allowed_extensions = {'mp3', 'jpg', 'jpeg', 'png'}

    # Get file extensions
    voice_extension = voice_filename.rsplit('.', 1)[1].lower() if '.' in voice_filename else ''
    image_extension = image_filename.rsplit('.', 1)[1].lower() if '.' in image_filename else ''

    # Ensure correct file types
    if voice_extension != 'mp3' or image_extension not in {'jpg', 'jpeg', 'png'}:
        return jsonify({'error': 'Invalid file types. Only MP3 and JPG/JPEG/PNG files are allowed.'}), 400

    # Save the files
    voice_file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], voice_filename)
    image_file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], image_filename)
    
    voice_file.save(voice_file_path)
    image_file.save(image_file_path)

    # Call the voice model's feature extraction function
    voice_result = voice_model_feature_extraction(voice_file_path)

    # Call the image model's feature extraction function
    image_result = image_model_feature_extraction(image_file_path)

    # Ensure the expected keys are in the results
    if 'prediction' not in voice_result or 'confidence' not in voice_result:
        return jsonify({'error': 'Invalid response from voice model'}), 500
    if 'prediction' not in image_result or 'confidence' not in image_result:
        return jsonify({'error': 'Invalid response from image model'}), 500

    # Extract predicted classes and confidence from both results
    voice_class = voice_result['prediction']
    voice_confidence = voice_result['confidence']

    image_class = image_result['prediction']
    image_confidence = image_result['confidence']

    # Determine the final result
    if voice_class == image_class:
        final_result = {
            'final_prediction': voice_class,
            'source': 'Both models agreed'
        }
    else:
        if voice_confidence >= image_confidence:
            final_result = {
                'final_prediction': voice_class,
                'source': 'Voice model (higher confidence)',
                'voice_confidence': voice_confidence,
                'image_confidence': image_confidence
            }
        else:
            final_result = {
                'final_prediction': image_class,
                'source': 'Image model (higher confidence)',
                'voice_confidence': voice_confidence,
                'image_confidence': image_confidence
            }

    return jsonify(final_result), 200
