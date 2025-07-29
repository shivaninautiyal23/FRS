import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
image_dir = "Fashion_Recommendation_System/static/images"  # or wherever your dataset images are
df = pd.read_csv("Fashion_Recommendation_System/fashion_products_sample.csv")  # or original dataset

def get_image_embedding(img_path):
    try:
        full_path = os.path.join(image_dir, img_path)
        img = tf.keras.preprocessing.image.load_img(full_path, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.resnet50.preprocess_input(x)
        features = model.predict(x)
        return features.flatten()
    except Exception as e:
        print("❌ Image load error:", img_path, "|", e)
        return np.zeros((2048,))

embeddings = []
for idx, row in df.iterrows():
    emb = get_image_embedding(row['image_path'])
    embeddings.append(emb)

df['image_embedding'] = embeddings

with open("fashion_data.pkl", "wb") as f:
    pickle.dump(df, f)

print("✅ New Embeddings Generated and Saved!")
