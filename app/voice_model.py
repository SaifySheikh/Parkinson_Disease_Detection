import numpy as np
from keras.models import load_model
import os
import librosa
import pandas as pd

# Define the base directory for models
base_dir = 'app/models'

# Load pre-trained models
lstm_gru_model = load_model(os.path.join(base_dir, 'lstm_gru_model.keras'))
cnn_lstm_model = load_model(os.path.join(base_dir, 'cnn_lstm_model.keras'))
xgb_nn_model = load_model(os.path.join(base_dir, 'xgb_nn_model.keras'))
xgb_nn_undersampler_model = load_model(os.path.join(base_dir, 'xgb_nn_undersampler_model.keras'))

def process_voice_file(features):
    # Drop non-numeric features before reshaping
    numeric_features = features.select_dtypes(include=[np.number])
    
    # Reshape for lstm_gru_model and cnn_lstm_model
    input_data = numeric_features.values.reshape(1, 1, -1)  # Reshape to (1, 1, number_of_features)
    cnn_input_data = numeric_features.values.reshape(1, 22, 1)  # Adjust if necessary
    xgb_input_data = numeric_features.values.reshape(1, -1)  # For XGBoost models

    # Make predictions using the models
    prediction1 = lstm_gru_model.predict(input_data)
    prediction2 = cnn_lstm_model.predict(cnn_input_data)
    prediction3 = xgb_nn_model.predict(xgb_input_data)
    prediction4 = xgb_nn_undersampler_model.predict(xgb_input_data)

    # Extract the predicted classes and their confidence scores
    predicted_class1 = np.argmax(prediction1, axis=1)
    predicted_class2 = np.argmax(prediction2, axis=1)
    predicted_class3 = np.argmax(prediction3, axis=1)
    predicted_class4 = np.argmax(prediction4, axis=1)

    # Combine predictions and calculate confidence scores
    predictions = [predicted_class1[0], predicted_class2[0], predicted_class3[0], predicted_class4[0]]
    confidence_scores = [np.max(prediction1), np.max(prediction2), np.max(prediction3), np.max(prediction4)]

    # Define labels based on predicted classes
    label_mapping = {0: "Healthy", 1: "Parkinson"}  # Adjust if your classes are different

    # Count predictions
    healthy_count = predictions.count(0)
    parkinson_count = predictions.count(1)

    # Initialize the final result variable
    final_result = None
    final_confidence = None

    # Decision logic
    if healthy_count >= 3:
        final_result = "Healthy"
        final_confidence = float(np.max([conf for pred, conf in zip(predictions, confidence_scores) if pred == 0]))
    elif parkinson_count >= 3:
        final_result = "Parkinson"
        final_confidence = float(np.max([conf for pred, conf in zip(predictions, confidence_scores) if pred == 1]))
    else:
        # If there is a tie (2 Healthy, 2 Parkinson), choose the one with the higher confidence
        if healthy_count == 2 and parkinson_count == 2:
            if confidence_scores[predictions.index(0)] > confidence_scores[predictions.index(1)]:
                final_result = "Healthy"
                final_confidence = float(confidence_scores[predictions.index(0)])
            else:
                final_result = "Parkinson"
                final_confidence = float(confidence_scores[predictions.index(1)])

    # Final results
    result = {
        'prediction': final_result,
        'confidence': final_confidence
    }

    return result

def voice_model_feature_extraction(file_path):
    y, sr = librosa.load(file_path, sr=None)
    features = {}

    # Feature extraction
    features['name'] = file_path.split('\\')[-1]  # Extracting the file name from the path
    features['MDVP:Fo(Hz)'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))  # Ensure float
    features['MDVP:Fhi(Hz)'] = float(np.max(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features['MDVP:Flo(Hz)'] = float(np.min(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features['MDVP:Jitter(%)'] = float(np.random.uniform(0, 1))  # Placeholder for actual calculation
    features['MDVP:Jitter(Abs)'] = float(np.random.uniform(0, 1))
    features['MDVP:RAP'] = float(np.random.uniform(0, 1))
    features['MDVP:PPQ'] = float(np.random.uniform(0, 1))
    features['Jitter:DDP'] = float(np.random.uniform(0, 1))
    features['MDVP:Shimmer'] = float(np.random.uniform(0, 1))
    features['MDVP:Shimmer(dB)'] = float(np.random.uniform(0, 1))
    features['Shimmer:APQ3'] = float(np.random.uniform(0, 1))
    features['Shimmer:APQ5'] = float(np.random.uniform(0, 1))
    features['MDVP:APQ'] = float(np.random.uniform(0, 1))
    features['Shimmer:DDA'] = float(np.random.uniform(0, 1))
    features['NHR'] = float(np.random.uniform(0, 1))
    features['HNR'] = float(np.random.uniform(0, 1))
    features['status'] = 'unknown'  # Placeholder
    features['RPDE'] = float(np.random.uniform(0, 1))
    features['DFA'] = float(np.random.uniform(0, 1))
    features['spread1'] = float(np.random.uniform(0, 1))
    features['spread2'] = float(np.random.uniform(0, 1))
    features['D2'] = float(np.random.uniform(0, 1))
    features['PPE'] = float(np.random.uniform(0, 1))

    features_df = pd.DataFrame([features])

    # Process the voice file and make predictions
    answer = process_voice_file(features_df)
    print(answer)

    return answer
