import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import pickle
from difflib import get_close_matches
# import os
# import ast


df = pickle.load(open('Fashion_Recommendation_System/fashion_data.pkl', 'rb'))



class FashionRecommender:
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.df = pickle.load(f)

        # Combine for TF-IDF
        self.df['combined'] = self.df[['product_name', 'category', 'gender', 'brand']].fillna('').agg(' '.join, axis=1)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.text_features = self.vectorizer.fit_transform(self.df['combined'])

        # Image model
        self.model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')

        # Ensure image_embedding are valid NumPy arrays
        if 'image_embedding' in self.df.columns and not isinstance(self.df['image_embedding'].iloc[0], np.ndarray):
            self.df['image_embedding'] = self.df['image_embedding'].apply(np.array)
        
    
    def get_image_embedding(self, img_path):
        try:
            print("Generating embedding for:", img_path)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = tf.keras.applications.resnet50.preprocess_input(x)
            features = self.model.predict(x)
            if np.sum(features) == 0:
                print("⚠️ Warning: All zero embedding for", img_path)
            return features.flatten()
        except Exception as e:
            print("❌ ERROR loading image:", e)
            return np.zeros((2048,))

    def recommend_by_name(self, product_name, filters=None, top_n=10):
        product_name_lower = product_name.lower()

        # Match by partial name or category
        matches = self.df[
            self.df['product_name'].str.lower().str.contains(product_name_lower, na=False) |
            self.df['category'].str.lower().str.contains(product_name_lower, na=False)
        ]


        if matches.empty:
            # Fuzzy fallback
            all_names = self.df['product_name'].dropna().str.lower().unique()
            close = get_close_matches(product_name_lower, all_names, n=1, cutoff=0.4)
            if close:
                matched_name = close[0]
                product = self.df[self.df['product_name'].str.lower() == matched_name]
            else:
                return pd.DataFrame(), None
        else:
            product = matches.iloc[[0]]

        if product.empty or product.iloc[0]['image_embedding'] is None:
            return pd.DataFrame(), None

        # Extract embedding of matched product
        vector = product.iloc[0]['image_embedding']

        # Calculate similarity for all items
        self.df['similarity'] = self.df['image_embedding'].apply(
            lambda x: cosine_similarity([vector], [x])[0][0] if x is not None else 0
        )

        filtered_df = self.df.copy()

        # If filters are selected by user
        if filters and any(filters.values()):
            print("⚙️ Applying user filters...")
            if filters.get('category'):
                filtered_df = filtered_df[filtered_df['category'].isin(filters['category'])]
            if filters.get('brand'):
                filtered_df = filtered_df[filtered_df['brand'].isin(filters['brand'])]
            if filters.get('gender'):
                filtered_df = filtered_df[filtered_df['gender'].isin(filters['gender'])]
        else:
            # No filters → fallback to same category + gender as base product
            same_cat = product.iloc[0]['category']
            same_product = product.iloc[0]['product_name']
            filtered_df = filtered_df[
                (filtered_df['category'] == same_cat) | (filtered_df['product_name'] == same_product)
            ]
            print("⚠️ No filters selected → Using fallback (same category + gender)")

        print("🎯 Final filtered items:", len(filtered_df))
        results = filtered_df.sort_values(by='similarity', ascending=False).head(top_n)

        return results, vector


    def recommend_by_image(self, img_path, top_n=5): 
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(x, axis=0))
        vector = self.model.predict(x).flatten()

        all_vectors = np.stack(self.df['image_embedding'].values)
        sim = cosine_similarity([vector], all_vectors).flatten()
        indices = np.argsort(sim)[::-1][:top_n]

        return self.df.iloc[indices][['product_name', 'image_path', 'brand']], vector

    def recommend_combined(self, img_path, top_n=5, category=None, gender=None, brand=None,classi=None, colour = None, alpha=0.7):
        # 1. Load and process uploaded image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.keras.applications.resnet50.preprocess_input(np.expand_dims(x, axis=0))
        image_vector = self.model.predict(x).flatten()

        # 2. Apply filters
        df_filtered = self.df.copy()
        if category:
            df_filtered = df_filtered[df_filtered['category'].str.lower() == category.lower()]
        if gender:
            df_filtered = df_filtered[df_filtered['gender'].str.lower() == gender.lower()]
        if brand:
            df_filtered = df_filtered[df_filtered['brand'].str.lower() == brand.lower()]
        if classi:
            df_filtered = df_filtered[df_filtered['class'].str.lower() == classi.lower()]
        if colour:
            df_filtered = df_filtered[df_filtered['colour'].str.lower() == colour.lower()]

        # If nothing remains after filtering
        if df_filtered.empty:
            return pd.DataFrame(), image_vector

        # 3. Get corresponding image_embedding
        image_vectors = np.stack(df_filtered['image_embedding'].values)
        text_indices = df_filtered.index.tolist()
        text_vectors = self.text_features[text_indices]

        # 4. Compute cosine similarities
        image_sim = cosine_similarity([image_vector], image_vectors).flatten()
        text_sim = cosine_similarity(self.vectorizer.transform([" ".join([category or "", gender or "", brand or "" , classi or "", colour or ""])]), text_vectors).flatten()

        # 5. Combine similarities
        combined_sim = alpha * image_sim + (1 - alpha) * text_sim

        # 6. Get top N
        indices = np.argsort(combined_sim)[::-1][:top_n]
        final_results = df_filtered.iloc[indices][['product_name', 'image_path', 'brand', 'category', 'gender','class','colour']]
    
        return final_results, image_vector


    def recommend_by_user_history(self, user_history, top_n=5):
        if not user_history:
            return pd.DataFrame()

        all_sim_scores = np.zeros(len(self.df))
        for item in user_history:
            if item.get('type') == 'name':
                idx = self.df[self.df['product_name'].str.lower() == item['value'].lower()].index
                if not idx.empty:
                    sim = cosine_similarity(self.text_features[idx[0]], self.text_features).flatten()
                    all_sim_scores += sim
            elif item.get('type') == 'tf.keras.preprocessing.image':
                image_vector = np.array(item['vector'])
                sim = cosine_similarity([image_vector], np.stack(self.df['image_embedding'].values)).flatten()
                all_sim_scores += sim
        indices = np.argsort(all_sim_scores)[::-1][:top_n]
        return self.df.iloc[indices][['product_name', 'image_path', 'brand']]



