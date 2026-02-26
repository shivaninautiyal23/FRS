import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import pickle
import random
# Category predictor
from pred.cat import predict_category   # we fix handling inside this file

# Suppress TF logs
tf.get_logger().setLevel('ERROR')

# =====================================================================
# CONFIG
# =====================================================================
MODEL_PATH   = r"C:/Users/Rajesh/OneDrive/Desktop/PYTHON/frs/model.h5"
PRODUCTS_CSV = r"C:/Users/Rajesh/OneDrive/Desktop/PYTHON/frs/fashion_products_sample.csv"
CACHE_FILE   = r"C:/Users/Rajesh/OneDrive/Desktop/PYTHON/frs/embeddings_cache.pkl"

print("🔄 Loading compatibility model...")

try:
    model = load_model(MODEL_PATH)
    print("✅ Siamese model loaded.")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    model = None

# =====================================================================
# LOAD PRODUCT DATA
# =====================================================================
try:
    products_df = pd.read_csv(PRODUCTS_CSV, encoding='latin1')

    required_cols = ["product_id", "gender", "category", "image_path"]
    for col in required_cols:
        if col not in products_df.columns:
            raise ValueError(f"❌ CSV missing required column '{col}'")

    print(f"✅ Loaded {len(products_df)} products.")

except Exception as e:
    print(f"❌ CSV load error: {e}")
    products_df = pd.DataFrame(columns=["product_id", "gender", "category", "image_path"])

# =====================================================================
# NORMALIZATION HELPERS
# =====================================================================
def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).lower().strip().replace(" ", "_").replace("-", "_")


def map_category(cat):
    if not isinstance(cat, str):
        return "other"

    cat = cat.lower()

    TOP = ["tshirt", "shirt", "hoodie", "sweater", "kurti", "top", "blouse", "jacket"]
    BOTTOM = ["jeans", "pants", "trouser", "trousers", "joggers",
              "leggings", "shorts", "skirt", "bottom"]

    if cat in TOP:
        return "topwear"
    if cat in BOTTOM:
        return "bottomwear"
    return "other"

products_df["category_norm"] = products_df["category"].apply(map_category)
products_df["gender_norm"] = products_df["gender"].str.lower().str.strip()


ALL_CATEGORIES = sorted(set(products_df["category_norm"].tolist()))
ALL_GENDERS    = ["men", "women", "unisex"]

# =====================================================================
# LOAD OR INIT CACHE
# =====================================================================
if os.path.exists(CACHE_FILE):
    try:
        with open(CACHE_FILE, "rb") as f:
            embeddings_cache = pickle.load(f)
        print(f"📦 Loaded embeddings cache: {len(embeddings_cache)} items")
    except:
        print("⚠️ Cache corrupted. Starting fresh.")
        embeddings_cache = {}
else:
    embeddings_cache = {}
    print("🆕 No cache found. Fresh start.")

# =====================================================================
# FEATURE EXTRACTOR
# =====================================================================
resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
base = "C:/Users/Rajesh/OneDrive/Desktop/PYTHON/frs/static/images"

# products_df["image_path"] = products_df["image_name"].apply(
#     lambda x: os.path.join(base, x)
# )

def extract_image_feature(path):
    """2048-dim feature extraction with caching."""
    # path = os.path.abspath(path)
    if not os.path.isabs(path):
        path = os.path.join(base, path)

    if path in embeddings_cache:
        return embeddings_cache[path]

    if not os.path.exists(path):
        print(f"⚠️ Missing image: {path}")
        return None

    try:
        img = image.load_img(path, target_size=(224, 224))
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        feat = resnet.predict(arr, verbose=0).flatten()
        embeddings_cache[path] = feat
        return feat

    except Exception as e:
        print(f"⚠️ Extraction failed for {path}: {e}")
        return None

# Atomic cache save
def save_cache():
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(embeddings_cache, f)
    os.replace(tmp, CACHE_FILE)

# =====================================================================
# RULE-BASED CATEGORY MATCHING
# =====================================================================
# category_mapping = {
#     "t_shirt": ["jeans", "shorts", "skirt", "pants", "trousers", "jacket", "footwear"],
#     "shirt":   ["jeans", "pants", "trousers", "skirt", "footwear", "jacket"],
#     "dress":   ["footwear", "jacket"],
#     "hoodie":  ["jeans", "pants", "shorts", "footwear"],
#     "jacket":  ALL_CATEGORIES,
#     "jeans":   ["t_shirt", "shirt", "hoodie", "jacket"],
#     "footwear": ALL_CATEGORIES,
#     "pants":    ["t_shirt", "shirt", "jacket"],
#     "skirt":    ["t_shirt", "shirt", "jacket"],
#     "trousers": ["shirt", "jacket"]
# }


# =====================================================================
# FIXED CATEGORY HANDLING FROM category_pred
# =====================================================================
def normalize_predicted_category(raw):
    """
    raw can be:
    - "tshirt"
    - ("tshirt", 0.67)
    - or some random word if ImageNet fails
    """

    if isinstance(raw, (tuple, list)):
        raw = raw[0]

    if raw is None:
        return ""

    raw = normalize_text(raw)

    # Map some common ImageNet outputs to your normalized vocabulary
    mapping = {
        "tshirt": "t_shirt",
        "t_shirt": "t_shirt",
        "jean": "jeans",
        "running_shoe": "footwear",
        "sandal": "footwear",
        "miniskirt": "skirt",
        "gown": "dress",
        "trench_coat": "jacket",
        "coat": "jacket",
        "shirt": "shirt",
    }

    if raw in mapping:
        return mapping[raw]

    return raw     # fallback

# =====================================================================
# COMPATIBILITY (2 images)
# =====================================================================
# def predict_compatibility_image_only(img1, img2):

#     if model is None:
#         raise RuntimeError("❌ Model not loaded")

#     # ---- FIX: normalize category output ----
#     cat1 = normalize_predicted_category(predict_category(img1))
#     cat2 = normalize_predicted_category(predict_category(img2))

#     VALID_PAIRS = {
#         ("t_shirt", "jeans"),
#         ("shirt", "jeans"),
#         ("dress", "footwear"),
#         ("t_shirt", "footwear"),
#         ("hoodie", "jeans"),
#         ("t_shirt", "pants"),
#         ("shirt", "pants"),
#         ("topwear", "bottomwear"),   # keep your old pairs too
#         ("bottomwear", "topwear"),
#     }

#     if (cat1, cat2) not in VALID_PAIRS and (cat2, cat1) not in VALID_PAIRS:
#         return 0.0   # fallback score for incompatible categories

#     # ---- Extract features ----
#     f1 = extract_image_feature(img1)
#     f2 = extract_image_feature(img2)

#     if f1 is None or f2 is None:
#         return 0.0

#     X = np.concatenate([f1, f2]).reshape(1, -1)

#     try:
#         score = float(model.predict(X, verbose=0)[0][0])
#     except Exception as e:
#         print("❌ Model prediction failed:", e)
#         return 0.0

#     return score



def predict_compatibility_image_only(img1, img2):

    if model is None:
        raise RuntimeError("Model not loaded")

    cat1 = predict_category(img1)
    cat2 = predict_category(img2)

    VALID = {
        ("topwear", "bottomwear"),
        ("bottomwear", "topwear")
    }

    if (cat1, cat2) not in VALID:
        return 0.0

    f1 = extract_image_feature(img1)
    f2 = extract_image_feature(img2)

    if f1 is None or f2 is None:
        return 0.0

    X = np.concatenate([f1, f2]).reshape(1, -1)
    score = float(model.predict(X, verbose=0)[0][0])

    return score

# =====================================================================
# RECOMMENDATION SYSTEM
# =====================================================================
# def get_recommendations_for_item(uploaded_image, gender, category, top_n=5):

#     if model is None:
#         raise RuntimeError("❌ Model not loaded.")

#     if products_df.empty:
#         raise RuntimeError("❌ Products dataset empty.")

#     # ---------------------------
#     # Normalize inputs
#     # ---------------------------
#     gender = normalize_text(gender)
#     category = normalize_text(category)     # "topwear" or "bottomwear"
    
#     # ---------------------------
#     # Extract features
#     # ---------------------------
#     f_upload = extract_image_feature(uploaded_image)
#     if f_upload is None:
#         raise ValueError("❌ Could not extract features from uploaded image.")

#     # ---------------------------
#     # Opposite category selection
#     # ---------------------------
#     if category == "topwear":
#         target_category = "bottomwear"
#     elif category == "bottomwear":
#         target_category = "topwear"
#     else:
#         # fallback (rare)
#         target_category = "bottomwear"

#     # ---------------------------
#     # Filter dataset
#     # ---------------------------
#     candidates = products_df[
#         (products_df["category_norm"] == target_category) &
#         (products_df["gender_norm"].isin([gender, "unisex"]))
#     ].copy()

#     print(products_df["category_norm"].unique())
#     print(products_df["gender_norm"].unique())
#     print("Gender coming:", gender)
#     print("Category coming:", category)
#     print("Target category:", target_category)

#     if len(candidates) == 0:
#         print("⚠️ No compatible products found.")
#         return []

#     # ---------------------------
#     # Extract candidate features
#     # ---------------------------
#     valid_rows = []
#     candidate_features = []

#     for _, row in candidates.iterrows():
#         feat = extract_image_feature(row["image_path"])
#         if feat is None:
#             continue
#         candidate_features.append(feat)
#         valid_rows.append(row)

#     if len(candidate_features) == 0:
#         print("⚠️ Candidate features missing.")
#         return []

#     candidate_features = np.array(candidate_features)
#     upload_rep = np.tile(f_upload, (len(candidate_features), 1))

#     # Siamese model expects concatenated features
#     X = np.concatenate([upload_rep, candidate_features], axis=1)

#     # ---------------------------
#     # Predict compatibility scores
#     # ---------------------------
#     try:
#         scores = model.predict(X, verbose=0).flatten()
#     except Exception as e:
#         print("❌ Prediction failed:", e)
#         return []

#     # ---------------------------
#     # Build sorted response
#     # ---------------------------
#     results = []
#     for row, score in zip(valid_rows, scores):
#         results.append({
#             "product_id": row["product_id"],
#             "category": row["category"],
#             "gender": row["gender"],
#             "score": round(float(score) * 100, 2),
#             "image_path": row["image_path"]
#         })

#     results = sorted(results, key=lambda x: x["score"], reverse=True)

#     save_cache()

#     return results[:top_n]

def get_recommendations_for_item(uploaded_image, gender, category, top_n=5):

    if model is None:
        raise RuntimeError("❌ Model not loaded.")

    if products_df.empty:
        raise RuntimeError("❌ Products dataset empty.")

    # ---------------------------
    # Normalize inputs
    # ---------------------------
    gender = normalize_text(gender)
    category = map_category(category)  # User input mapped
    print("User category mapped to:", category)

    # ---------------------------
    # Extract features from uploaded image
    # ---------------------------
    f_upload = extract_image_feature(uploaded_image)
    if f_upload is None:
        raise ValueError("❌ Could not extract features from uploaded image.")

    # ---------------------------
    # Opposite category
    # ---------------------------
    if category == "topwear":
        target_category = "bottomwear"
    elif category == "bottomwear":
        target_category = "topwear"
    else:
        target_category = "bottomwear"

    print("Target category:", target_category)

    # ---------------------------
    # Filter candidates
    # ---------------------------
    candidates = products_df[
        (products_df["category_norm"] == target_category) &
        (products_df["gender_norm"].isin([gender, "unisex"]))
    ].copy()

    print("Candidates found:", len(candidates))

    if len(candidates) == 0:
        print("⚠️ No compatible products found after mapping.")
        return []

    # ---------------------------
    # Extract candidate features
    # ---------------------------
    valid_rows = []
    candidate_features = []

    for _, row in candidates.iterrows():
        feat = extract_image_feature(row["image_path"])
        if feat is None:
            continue
        candidate_features.append(feat)
        valid_rows.append(row)

    if len(candidate_features) == 0:
        print("⚠️ No features extracted from candidate images.")
        return []

    candidate_features = np.array(candidate_features)
    upload_rep = np.tile(f_upload, (len(candidate_features), 1))

    # Siamese model expects concatenated features
    X = np.concatenate([upload_rep, candidate_features], axis=1)

    # ---------------------------
    # Predict compatibility scores
    # ---------------------------
    try:
        scores = model.predict(X, verbose=0).flatten()
    except Exception as e:
        print("❌ Prediction failed:", e)
        return []

    # ---------------------------
    # Build response
    # ---------------------------
    results = []
    for row, score in zip(valid_rows, scores):
        results.append({
            "product_id": row["product_id"],
            "category": row["category"],
            "gender": row["gender"],
            "score": round(float(score) * 100, 2),
            "image_path": row["image_path"]
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)

    save_cache()

    random_results = random.sample(results, min(top_n, len(results)))
    return random_results

