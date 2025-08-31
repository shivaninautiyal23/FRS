import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import pickle
from difflib import get_close_matches
import difflib
from keras.utils import register_keras_serializable
import re

gender_keywords = {
    "women": ["women", "woman", "female", "girl", "lady", "gal"],
    "men": ["men", "man", "male", "boy", "guy", "gentleman"],
    "unisex": ["unisex", "everyone", "all", "any","boy"]
}

all_categories = [
    "T-Shirt", "tshirt", "T shirt" "Jeans", "Shirt", "Dress", "Hoodie", "Skirt", "Jacket", "Boots",
    "Top", "Pants", "Belt", "Socks", "Shoes", "Blazer", "Top Kurti", "Trousers",
    "Shorts", "Joggers", "Cargos", "Frock", "Bra", "Short Kurti", "Watch",
    "Bracelet", "Pendant", "Sunglass", "Ring", "Pant", "Cargo", "Sweatpants",
    "Capri", "Trackpants", "Footwear"
]

# Category compatibility mapping (looser combos)
category_mapping = {
    "T-Shirt": ["Jeans", "Shorts", "Skirt", "Pants", "Trousers", "Cargo", "Joggers", "Trackpants"],
    "Shirt": ["Jeans", "Trousers", "Skirt", "Pants", "Shorts"],
    "Dress": ["Shoes", "Boots", "Sunglass", "Bracelet", "Pendant", "Belt"],
    "Hoodie": ["Jeans", "Joggers", "Sweatpants", "Trackpants"],
    "Skirt": ["Top", "T-Shirt", "Shirt", "Hoodie"],
    "Jacket": ["T-Shirt", "Shirt", "Dress", "Top"],
    "Boots": ["Jeans", "Dress", "Skirt", "Pants"],
    "Top": ["Jeans", "Skirt", "Shorts", "Pants", "Trousers"],
    "Pants": all_categories,
    "Belt": all_categories,
    "Socks": all_categories,
    "Shoes": all_categories,
    "Blazer": all_categories,
    "Top Kurti": ["Jeans", "Trousers", "Pants"],
    "Trousers": all_categories,
    "Shorts": ["T-Shirt", "Top", "Shirt", "Hoodie"],
    "Jeans": all_categories,
    "Joggers": all_categories,
    "Cargos": all_categories,
    "Frock": ["Shoes", "Boots", "Belt", "Sunglass"],
    "Bra": all_categories,
    "Short Kurti": ["Jeans", "Trousers", "Pants"],
    "Watch": all_categories,
    "Bracelet": all_categories,
    "Pendant": all_categories,
    "Sunglass": all_categories,
    "Ring": all_categories,
    "Pant": all_categories,
    "Cargo": all_categories,
    "Sweatpants": all_categories,
    "Capri": all_categories,
    "Trackpants": all_categories,
    "Footwear": all_categories
}

@register_keras_serializable()
def l2_normalize(x):
    return tf.math.l2_normalize(x, axis=1)

# Load the model with custom_objects
embedding_model = tf.keras.models.load_model(
    "C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/siamese_embedding_model_tf.h5",
    custom_objects={"l2_normalize": l2_normalize}
)

# Load dataset
with open("C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/fashion_data_with_siamese.pkl", "rb") as f:
    df = pickle.load(f)

# Preloaded embeddings (so we don’t calculate each time)
with open("C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/fashion_data_with_siamese.pkl", "rb") as f:
    product_embeddings = pickle.load(f)  # numpy array of embeddings

# Function: Fuzzy text match
def fuzzy_match(query, choices, threshold=0.7):
    matches = difflib.get_close_matches(query, choices, n=10, cutoff=threshold)
    return matches

def recommend_by_search(query):
    query = query.strip().lower()
    top_n = 50

    gender = None
    category = None

    # Detect gender (whole word match)
    for g, keywords in gender_keywords.items():
        for word in keywords:
            if isinstance(word, str) and re.search(rf"\b{re.escape(word.lower())}\b", query):
                gender = g
                break
        if gender:
            break

    # Detect category (whole word match)
    for cat in all_categories:
        if isinstance(cat, str) and re.search(rf"\b{re.escape(cat.lower())}\b", query):
            category = cat
            break

    print(str(category) + " and " + str(gender))

    # Filter DataFrame
    filtered_df = df.copy()
    if gender:
        filtered_df = filtered_df[filtered_df['gender'].str.lower() == gender.lower()]
    if category:
        filtered_df = filtered_df[filtered_df['category'].str.lower() == category.lower()]

    if filtered_df.empty:
        return []
    
    filtered_df = filtered_df.reset_index()

    product_names = filtered_df['product_name'].str.lower().tolist()

    # Step 0: Exact match (highest priority)
    exact_matches = filtered_df[filtered_df['product_name'].str.lower() == query]
    if not exact_matches.empty:
        return exact_matches.head(top_n).to_dict(orient="records")

    # Step 1: Fuzzy match
    matches = fuzzy_match(query, product_names)
    if matches:
        matched_indices = filtered_df[filtered_df['product_name'].str.lower().isin(matches)].index
        return filtered_df.loc[matched_indices].head(top_n).to_dict(orient="records")
    


    closest_name = difflib.get_close_matches(query, product_names, n=1, cutoff=0.0)
    if not closest_name:
        return []

    matched_rows = filtered_df[filtered_df['product_name'].str.lower() == closest_name[0]]
    if matched_rows.empty:
        matched_rows = filtered_df[filtered_df['product_name'].str.lower().str.contains(re.escape(closest_name[0]))]
        if matched_rows.empty:
            return []
    query_idx = matched_rows.index[0]  # position in filtered_df_reset
    original_idx = filtered_df.loc[query_idx, 'index'] 
    # query_idx = matched_rows.first_valid_index()
    # if query_idx is None:
    #     return []

   # query_embedding = product_embeddings[original_idx].reshape(1, -1)
    query_embedding = product_embeddings[original_idx]
    filtered_embeddings = product_embeddings[filtered_df['index'].tolist()]

    sims = cosine_similarity(query_embedding, filtered_embeddings)[0]
    top_indices = np.argsort(sims)[::-1][1:top_n+1]

    return filtered_df.iloc[top_indices].to_dict(orient="records")

# Function: Get embedding for product image
def get_embedding(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    embedding = embedding_model.predict(img_array)
    return embedding.flatten()

class FashionRecommender:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.df = pickle.load(f)

        # Combine relevant text columns for TF-IDF vectorizer
        self.df['combined'] = self.df[['product_name', 'category', 'gender', 'brand']].fillna('').agg(' '.join, axis=1)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.text_features = self.vectorizer.fit_transform(self.df['combined'])

        # Load pretrained ResNet50 (image feature extractor)
        self.model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

        # Fix siamese_embedding column (make sure it contains numpy arrays)
        if 'siamese_embedding' in self.df.columns and not isinstance(self.df['siamese_embedding'].iloc[0], np.ndarray):
            self.df['siamese_embedding'] = self.df['siamese_embedding'].apply(np.array)

        # Embeddings dict for quick lookup (for compatibility checker etc)
        self.embeddings_dict = {
            row['product_id']: row['siamese_embedding']
            for _, row in self.df.iterrows()
            if 'siamese_embedding' in row and isinstance(row['siamese_embedding'], (list, np.ndarray))
        }
    def get_siamese_embedding(self, img_path):
        try:
            print("Generating embedding for:", img_path)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.resnet50.preprocess_input(x)
            features = self.model.predict(x)
            return features.flatten()
        except Exception as e:
            print("❌ ERROR loading image:", e)
            return np.zeros((2048,))

    def get_product_by_id(self, product_id):
        row = self.df[self.df['product_id'] == product_id]
        return row.iloc[0].to_dict() if not row.empty else None

    def recommend_by_image(self, img_path, top_n=5):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(x, axis=0))
        vector = self.model.predict(x).flatten()

        all_vectors = np.stack(self.df['siamese_embedding'].values)
        sim = cosine_similarity([vector], all_vectors).flatten()
        indices = np.argsort(sim)[::-1][:top_n]

        return self.df.iloc[indices][['product_name', 'image_path', 'brand']], vector

'''recommendation k lie function alg se'''
    
def recommend_items(product_name, gender=None, top_n=10):
    # Normalize product name for search
    product_name = product_name.strip().lower()

    # Find the query product
    query_row = df[df['product_name'].str.lower() == product_name]
    if query_row.empty:
        raise ValueError(f"Product '{product_name}' not found in dataset.")

    # Extract query details
    query_embedding = np.array(query_row['siamese_embedding'].iloc[0]).reshape(1, -1)
    query_category = query_row['category'].iloc[0]
    query_gender = query_row['gender'].iloc[0]

    # Use gender from argument if provided, else use product's gender
    selected_gender = gender.strip().lower() if gender else query_gender.lower()

    # Ensure category exists in mapping
    if query_category not in category_mapping:
        raise ValueError(f"Category '{query_category}' not in mapping.")

    allowed_categories = category_mapping[query_category]

    # Filter dataset by allowed categories + gender
    filtered_df = df[
        (df['category'].isin(allowed_categories)) &
        (df['gender'].str.lower() == selected_gender)
    ].copy()

    if filtered_df.empty:
        raise ValueError(f"No products found matching category '{query_category}' and gender '{selected_gender}'.")

    # Calculate cosine similarity
    sims = cosine_similarity(query_embedding, np.stack(filtered_df['siamese_embedding'].to_numpy()))[0]
    filtered_df['similarity'] = sims

    # Sort and select top N
    results = filtered_df.sort_values(by='similarity', ascending=False).head(top_n)

    # Return as list of dictionaries for easy template rendering
    return results[['product_name', 'category', 'gender', 'image_path']].to_dict(orient='records')


# load precomputed image embeddings
EMB_PKL = "C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/image_embeddings.pkl"  # from compute_embeddings.py
emb_map = {}
try:
    emb_map = pickle.load(open(EMB_PKL, "rb"))
except:
    print("Warning: embeddings pkl not found at", EMB_PKL)

# load trained compatibility head
try:
    compat_model = tf.keras.models.load_model("C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/compatibility_head.h5")
except Exception as e:
    compat_model = None
    print("compatibility model not loaded:", e)


def get_embedding_for_path(path):
    # path should match keys in emb_map (image_path or filename depending on your map)
    if path in emb_map:
        return emb_map[path]
    # Optionally, fallback compute with embedding_model:
    try:
        from rec import embedding_model, get_embedding  # adjust based on your code
        emb = get_embedding(path)
        return emb
    except Exception:
        return None

def compatibility_score(img_path_a, img_path_b):
    emb_a = get_embedding_for_path(img_path_a)
    emb_b = get_embedding_for_path(img_path_b)
    if emb_a is None or emb_b is None:
        return 0.0
    if compat_model:
        inp = np.concatenate([emb_a, emb_b]).reshape(1, -1)
        return float(compat_model.predict(inp)[0][0])  # between 0 and 1
    else:
        # fallback heuristics: cosine similarity of embeddings
        return float(cosine_similarity(emb_a.reshape(1,-1), emb_b.reshape(1,-1))[0][0])


def recommend_items_ml(product_name, gender=None, top_n=10, threshold=0.2):
    # find query product row (same as before)
    query_row = df[df['product_name'].str.lower() == product_name.strip().lower()]
    if query_row.empty:
        raise ValueError("Product not found")
    query_category = query_row['category'].iloc[0]
    selected_gender = gender.strip().lower() if gender else query_row['gender'].iloc[0].lower()

    if query_category not in category_mapping:
        raise ValueError("Category not in mapping")

    allowed = category_mapping[query_category]
    candidates = df[(df['category'].isin(allowed)) & (df['gender'].str.lower() == selected_gender)].copy()
    if candidates.empty:
        return []

    # For each candidate compute compatibility score with query product
    query_image = query_row['image_path'].iloc[0]
    scores = []
    for _, r in candidates.iterrows():
        cand_image = r['image_path']
        score = compatibility_score(query_image, cand_image)
        scores.append(score)

    candidates['compat_score'] = scores
    candidates = candidates.sort_values(by='compat_score', ascending=False)
    # filter by threshold optionally
    candidates = candidates[candidates['compat_score'] >= threshold]
    top = candidates.head(top_n)
    return top[['product_name','image_path','category','compat_score']].to_dict(orient='records')
