# from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.preprocessing import image
# import numpy as np
# import os
# import tensorflow as tf

# # Suppress TensorFlow warnings
# tf.get_logger().setLevel('ERROR')

# # Initialize ResNet50 model for feature extraction
# # We use include_top=False to only get the feature layers, and pooling='avg' to get a fixed-size vector.
# model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# def extract_image_feature(image_path):
#     """
#     Loads an image, preprocesses it, and extracts a 2048-dimensional feature vector
#     using the pre-trained ResNet50 model.
#     """
#     try:
#         # ResNet50 expects images of size (224, 224)
#         img = image.load_img(image_path, target_size=(224, 224))
#         x = image.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         # Preprocess input (e.g., scale pixel values)
#         x = preprocess_input(x)
#         # Predict features and flatten the result (2048 features)
#         feature = model.predict(x, verbose=0).flatten()
#         return feature
#     except Exception as e:
#         print(f"Error processing image {image_path}: {e}")
#         return None

# def build_feature_database(products_df, images_folder):
#     """
#     Builds a dictionary of features for all products.
#     The key is the image filename (e.g., 'img001.jpg').
#     NOTE: This process can be slow. In production, load features from a precomputed .npy file.
#     """
#     features = {}
#     # IMPORTANT: The correct column name for the image file path is 'image_path' based on the CSV.
#     for idx, row in products_df.iterrows():
#         img_file = row['image_path'].strip()
#         img_path = os.path.join(images_folder, img_file)
#         if os.path.exists(img_path):
#             feature = extract_image_feature(img_path)
#             if feature is not None:
#                  # Store feature using the image filename as key
#                 features[img_file] = feature
#         else:
#              print(f"Warning: Image file not found at {img_path}")
#     return features

from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras import models
import numpy as np
import os
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Load the ResNet50 model (pre-trained on ImageNet, without the classification layer)
# The 'pooling'='avg' option returns a 2048-dimensional feature vector per image.
# NOTE: The compatibility model is trained on the concatenation of two features (4096 dimensions).
base_model = models.Sequential([
    ResNet50(weights='imagenet', include_top=False, pooling='avg')
])

def extract_image_feature(image_path):
    """
    Loads an image, preprocesses it for ResNet50, and extracts the feature vector.
    
    Args:
        image_path (str): The local path to the image file.
        
    Returns:
        np.array: A 2048-dimensional feature vector, or None if processing fails.
    """
    try:
        # Load image and resize to 224x224 (required by ResNet50)
        img = image.load_img(image_path, target_size=(224, 224))
        # Convert to numpy array
        x = image.img_to_array(img)
        # Add batch dimension
        x = np.expand_dims(x, axis=0)
        # Preprocess input (scaling pixel values according to ResNet50 training)
        x = preprocess_input(x)
        
        # Extract features (output shape is (1, 2048) for ResNet50 with pooling='avg')
        feature = base_model.predict(x, verbose=0).flatten() 
        return feature
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def build_feature_database(products_df, images_folder):
    """
    Builds a dictionary of features for all products from the CSV.
    
    Args:
        products_df (pd.DataFrame): DataFrame containing product metadata.
        images_folder (str): Base path to the image directory.
        
    Returns:
        dict: A dictionary mapping image filenames (e.g., 'img001.jpg') to feature vectors.
    """
    features = {}
    for idx, row in products_df.iterrows():
        # Using 'image_path' column name based on the sample CSV structure
        img_file = row['image_path'] 
        img_path = os.path.join(images_folder, img_file)
        feature = extract_image_feature(img_path)
        if feature is not None:
             features[img_file] = feature
    return features
