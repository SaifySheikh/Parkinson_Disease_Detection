import numpy as np
from keras.models import load_model
import os
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder

# Define the base directory for models
base_dir = 'app/models'

# Load pre-trained CNN model
image_model1 = load_model(os.path.join(base_dir, 'cnn.h5'))

# Define class labels, e.g., 'healthy' and 'parkinson'
classes = ['healthy', 'parkinson']

# Label Encoder to convert class names to numbers
label_encoder = LabelEncoder()
label_encoder.fit(classes)

def preprocess_image(file_path, target_size, grayscale=False):
    """Preprocess the input image for the CNN model."""
    if grayscale:
        img = image.load_img(file_path, target_size=target_size, color_mode='grayscale')
    else:
        img = image.load_img(file_path, target_size=target_size)
        
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    return img_array

def image_model_feature_extraction(file_path):
    """Extract features and predict class from the input image."""
    
    # Preprocess the image for model1 (change size if necessary)
    processed_image1 = preprocess_image(file_path, (128, 128), grayscale=True)

    # Get predictions from model1
    predictions1 = image_model1.predict(processed_image1)

    # Check the shapes of the predictions
    print(f'Prediction 1 shape: {predictions1.shape}')

    # Decode the predicted classes
    predicted_class1 = label_encoder.inverse_transform([np.argmax(predictions1)])

    # Prepare results
    results = {
        'prediction': predicted_class1[0],
        'confidence': float(np.max(predictions1)),
    }

    # Print the results for debugging
    print(results)

    return results

# Call model summary to see the layers and the expected input/output shapes
print("Model 1 Summary:")
image_model1.summary()

# # Example usage:
# if __name__ == "__main__":
#     # Example image path (replace with an actual image path)
#     test_image_path = 'path/to/your/test/image.jpg'
#     results = image_model_feature_extraction(test_image_path)
