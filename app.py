from flask import Flask, jsonify, render_template, request, redirect, session, url_for, flash
from werkzeug.utils import secure_filename
import os
from  rec import FashionRecommender , recommend_items ,recommend_by_search, recommend_items_ml
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin ,LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
from keras.models import load_model
import tensorflow as tf



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/saved_items.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # route name

app.secret_key = 'secret123'
app.permanent_session_lifetime = timedelta(days=1)

UPLOAD_FOLDER = 'Fashion_Recommendation_System/static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Path to your combined pickle (metadata + embeddings)
dataframe_path = 'C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/fashion_data_with_siamese.pkl'

# Instantiate your recommender with just the combined pickle
recommender = FashionRecommender(dataframe_path)

# Load dataframe once for standalone recommend_items function
df = pd.read_pickle(dataframe_path)

# Load once at app startup
siamese_model = load_model("C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/siamese_embedding_model_tf.h5")
compatibility_model = load_model("C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/compatibility_head.h5")

class SavedItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_id = db.Column(db.String(100), nullable=False)  # Matches the CSV ID
    product_name = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(100))
    gender = db.Column(db.String(50))
    brand = db.Column(db.String(100))
    image_path = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('saved_items', lazy=True))

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

with app.app_context():
    db.create_all()
    print("DB created:", os.path.exists('C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/saved_items.db'))


products = df.to_dict(orient='records')


@app.route('/')
def home():
    return render_template('index.html')


# def get_product_by_id(self, product_id):
#     row = self.df[self.df['product_id'] == product_id]
    # return row.iloc[0] if not row.empty else None


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))
 

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)  # <-- this is the magic
            flash("Logged in successfully!")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.")
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # check if user already exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already exists. Try a different one.")
            return redirect(url_for('register'))

        # create user
        hashed_pw = generate_password_hash(password)
        new_user = User(username=username, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created! You can now log in.")
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route("/search", methods=["GET", "POST"])
def search():
    # Merge GET and POST data handling
    query = request.values.get("q", "").strip()

    if not query:
        results=[]
        return render_template("product.html", recommendations=results, query=query)

    results = recommend_by_search(query)
    return render_template("product.html", recommendations=results, query=query)


@app.route('/recommend/image', methods=['POST'])
def recommend_image():
    file = request.files['image']
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        results, vector = recommender.recommend_by_image(path)
        log_user_action('image', filename, vector)
        return render_template('product.html', recommendations=results.to_dict(orient='records'), uploaded_image=filename)

@app.route('/recommend/personal')
def recommend_personal():
    history = session.get('history', [])
    results = recommender.recommend_by_user_history(history)
    return render_template('product.html', recommendations=results.to_dict(orient='records'))

def log_user_action(type_, value, vector=None):
    session.permanent = True
    if 'history' not in session:
        session['history'] = []
    entry = {'type': type_, 'value': value, 'vector': vector.tolist() if vector is not None else None}
    session['history'].append(entry)
    session['history'] = session['history'][-10:]

@app.route('/category/<gender>')
def category_recommendation(gender):
    recommender = FashionRecommender('Fashion_Recommendation_System/fashion_data.pkl')
    
    # Example: fetch top 10 items from same category
    recommendations = recommender.df[
        recommender.df['gender'].str.strip().str.lower() == gender.strip().lower()
    ].head(10).to_dict(orient='records')
    
    print("Requested gender:", gender)
    print("Available genders in dataset:", recommender.df['gender'].unique())
    print("Filtered gender match count:",  len(recommender.df[recommender.df['gender'].str.strip().str.lower() == gender.strip().lower()]))

    return render_template(
        'product.html',
        recommendations=recommendations,
        uploaded_image=None,
        category_name=gender
    )

@app.route('/recommend/filters', methods=['GET', 'POST'])
def recommend_with_filters():
    if request.method == 'POST':
        selected_categories = request.form.getlist('category')
        selected_brands = request.form.getlist('brand')
        selected_gender = request.form.get('gender')

        # Apply filters
        df = recommender.df
        if selected_categories:
            df = df[df['category'].str.lower().isin([cat.lower() for cat in selected_categories])]
        if selected_brands:
            df = df[df['brand'].str.lower().isin([b.lower() for b in selected_brands])]
        if selected_gender:
            df = df[df['gender'].str.lower() == selected_gender.lower()]

        recommendations = df.head(10).to_dict(orient='records')

        return render_template('product.html', recommendations=recommendations)

    # For GET request: fetch all unique values to show in form
    categories = sorted(recommender.df['category'].dropna().unique())
    brands = sorted(recommender.df['brand'].dropna().unique())
    genders = sorted(recommender.df['gender'].dropna().unique())

    return render_template('filters.html', categories=categories, brands=brands, genders=genders)

@app.route('/outfit', methods=['GET'])
def outfit_form():
    return render_template('outfit_form.html')

@app.route('/outfits', methods=['GET'])
def outfit_forms():
    return render_template('outfit_form_score.html')

@app.route('/recommend/outfit', methods=['GET'])
def recommend_outfit():
    product_name = request.args.get('product_name')
    gender = request.args.get('gender')

    # Ensure recommend_items supports gender filtering
    recommendations = recommend_items(product_name, gender)

    return render_template(
        "recommendation.html",
        recommendations=recommendations,
        query=product_name,
        gender=gender
    )

# def preprocess_image(image_path):
#     img = Image.open(image_path).convert("RGB").resize((224, 224))
#     img_array = np.array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)

def get_embedding_from_image(image_path):
    img_batch = preprocess_image(image_path)
    embedding = compatibility_model.predict(img_batch)
    return embedding

def check_outfit_compatibility_images(img1_path, img2_path, threshold=-1):
    emb1 = get_embedding_from_image(img1_path)
    emb2 = get_embedding_from_image(img2_path)

    # Combine embeddings the same way you trained the compatibility model
    combined = np.concatenate([emb1, emb2], axis=-1)

    score = compatibility_model.predict(combined)[0][0]
    is_compatible = score >= threshold
    print("Raw score:", score)
    print("Threshold:", threshold)
    return score, is_compatible

@app.route('/recommend/Outfits', methods=['GET', 'POST'])
def recommend_outfits():
    if request.method == 'POST':
        img1 = request.files.get('image1')
        img2 = request.files.get('image2')

        if not img1 or not img2:
            return render_template("outfit_result.html", error="Please upload two images.")

        # Save uploaded files temporarily
        img1_path = os.path.join('C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/static/uploads', img1.filename)
        img2_path = os.path.join('C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/static/uploads', img2.filename)
        img1.save(img1_path)
        img2.save(img2_path)
        
        try:
            score, is_compatible = check_outfit_compatibility_images(img1_path, img2_path)
        except Exception as e:
            return render_template("recommendations.html", error=str(e))

        return render_template(
                                "recommendations.html",
                                image1_path=img1_path,
                                image2_path=img2_path,
                                score=round(float(score), 3),
                                is_compatible=is_compatible
                                )

    return render_template("outfit_form.html")

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    return img

def get_embedding(image_path):
    img = preprocess_image(image_path)
    embedding = siamese_model.predict(img)[0]  # shape: (256,)
    return embedding

def get_compatibility_score(image1_path, image2_path):
    emb1 = get_embedding(image1_path)
    emb2 = get_embedding(image2_path)
    merged = np.concatenate([emb1, emb2]).reshape(1, -1)  # shape: (1, 512)
    score = compatibility_model.predict(merged)[0][0]
    return float(score)

@app.route("/check_compatibility", methods=["POST"])
def check_compatibility():
    if request.method == 'POST':
        img1 = request.files.get('image1')
        img2 = request.files.get('image2')

        if not img1 or not img2:
            return render_template("outfit_result.html", error="Please upload two images.")
        
        img1_filename = img1.filename
        img2_filename = img2.filename

        # Save uploaded files temporarily
        img1_path = os.path.join('C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/static/uploads', img1.filename)
        img2_path = os.path.join('C:/Users/Rajesh/OneDrive/Desktop/PYTHON/Fashion_Recommendation_System/static/uploads', img2.filename)
        img1.save(img1_path)
        img2.save(img2_path)

        if not img1 or not img2:
            return render_template("outfit_result.html", error="Please upload two images.")
        score = get_compatibility_score(img1_path, img2_path)

        return render_template(
            "recommendations.html",
            image1=img1_filename,
            image2=img2_filename,
            score=round(score, 3),
            is_compatible=score >= 0.5
            )






@app.route('/like', methods=['POST'])
def like_product():
    product_id = request.form.get('product_id')
    
    # Save to a file/database for personalized history (basic example below)
    with open('liked_products.txt', 'a') as f:
        f.write(product_id + '\n')
    
    # Optionally redirect back to the previous page
    return redirect(request.referrer or url_for('index'))

@app.route('/save/<product_id>', methods=['POST'])
@login_required
def save_item(product_id):
    # Get product from dataframe
    product_row = recommender.df[recommender.df['product_id'] == product_id]
    
    if product_row.empty:
        return "Product not found", 404

    product = product_row.iloc[0]

    # Avoid duplicates for current user
    already_saved = SavedItem.query.filter_by(
        product_id=product['product_id'],
        user_id=current_user.id
    ).first()

    if already_saved:
        flash("This item is already saved!", "warning")
        return redirect(request.referrer)

    # Create and save new item
    saved_item = SavedItem(
        product_id=product['product_id'],
        product_name=product['product_name'],
        category=product['category'],
        gender=product['gender'],
        brand=product['brand'],
        image_path=product['image_path'],
        user_id=current_user.id
    )

    db.session.add(saved_item)
    db.session.commit()
    flash("Item saved successfully!", "success")
    return redirect(request.referrer)

    
@app.route('/saved')
@login_required
def view_saved():
    items = SavedItem.query.filter_by(user_id=current_user.id).all()
    return render_template('saved.html', saved_items=items)

@app.route('/remove/<int:item_id>', methods=['POST'])
def remove_item(item_id):
    item = SavedItem.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    return redirect(url_for('view_saved'))

if __name__ == '__main__':
    app.run(debug=True)
