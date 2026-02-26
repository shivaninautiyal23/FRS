import pandas as pd
import numpy as np
from preprocess import extract_image_feature
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# --- File Paths (Using relative paths for better portability) ---
# NOTE: Assume these files are in the same directory as this script, 
# and the images are in a subdirectory named 'images'.
pairing_csv = 'C:/Users/Rajesh/OneDrive/Desktop/PYTHON/frs/Pairing.csv'
products_csv = 'C:/Users/Rajesh/OneDrive/Desktop/PYTHON/frs/fashion_products_sample.csv'
images_folder = 'C:/Users/Rajesh/OneDrive/Desktop/PYTHON/frs/images'
model_path = 'C:/Users/Rajesh/OneDrive/Desktop/PYTHON/frs/model.h5'

# Load datasets with encoding to avoid decode errors
try:
    pairing_df = pd.read_csv(pairing_csv, encoding='latin1')
    products_df = pd.read_csv(products_csv, encoding='latin1')
except FileNotFoundError as e:
    print(f"Error: Required file not found. Ensure {e.filename} is in the correct directory.")
    exit()

# --- 1. Feature Extraction ---

def get_feature_database(df, img_folder):
    """Extracts features for all images listed in the product DataFrame."""
    features = {}
    missing_images = []
    
    # Ensure ResNet model is loaded in preprocess.py before calling extract_image_feature
    print("Extracting features (This may take a few minutes)...")
    for idx, row in df.iterrows():
        # Corrected column name to match fashion_products_sample.csv
        img_file = row['image_path']  
        full_img_path = os.path.join(img_folder, img_file)
        
        if os.path.exists(full_img_path):
            try:
                # The function returns a 2048-dim feature vector
                features[img_file] = extract_image_feature(full_img_path)
            except Exception as e:
                print(f"Error extracting features for {img_file}: {e}")
                missing_images.append(img_file)
        else:
            missing_images.append(img_file)
            
    print(f"Extraction complete. {len(features)} features extracted. {len(missing_images)} images skipped.")
    return features

features = get_feature_database(products_df, images_folder)


# --- 2. Data Preparation ---

def get_pair_features(row, features_dict):
    """Concatenates the features of two images in a pair."""
    img_a = row['img_a']
    img_b = row['img_b']
    
    # Use image filename as the key
    if img_a not in features_dict or img_b not in features_dict:
        # print(f"Missing feature for pair ({img_a}, {img_b}) - skipping")
        return None
        
    f1 = features_dict[img_a]
    f2 = features_dict[img_b]
    
    # Concatenate features: total dimension will be 2048 + 2048 = 4096
    return np.concatenate([f1, f2])

# Prepare training data safely, only including pairs where both features are available
pair_features = []
valid_labels = []

for _, row in pairing_df.iterrows():
    pair_vec = get_pair_features(row, features)
    if pair_vec is not None:
        pair_features.append(pair_vec)
        valid_labels.append(row['label'])

# Convert to numpy arrays
X = np.array(pair_features)
y = np.array(valid_labels)

print(f"Prepared training data shape: {X.shape}, Labels shape: {y.shape}")

# --- 3. Train/Test/Validation Split ---

# 1. Split into Training (80%) and Temp (20%)
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Split Temp into Training (60% of total) and Validation (20% of total)
# (0.25 * 0.8 = 0.20 or 20% of the original data)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.25, random_state=42, stratify=y_train_temp
)

print(f"\n--- Data Split Summary ---")
print(f"Training set (60%): {X_train.shape[0]} samples")
print(f"Validation set (20%): {X_val.shape[0]} samples")
print(f"Test set (20%): {X_test.shape[0]} samples")
print("--------------------------\n")


# --- 4. Model Definition and Training ---

input_dim = X.shape[1] # Should be 4096 (2048 * 2)

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Sigmoid for binary classification (0 or 1)
])

# Use Adam optimizer and Binary Crossentropy loss for compatibility prediction
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=50, # Increased epochs for better learning
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# Evaluate the model on the held-out test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Test Accuracy: {accuracy*100:.2f}%")

# Save the trained model
model.save(model_path)
print(f"Trained model saved successfully as {model_path}")
