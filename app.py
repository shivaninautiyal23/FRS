from flask import Flask, render_template, request, redirect, session, url_for, flash
from werkzeug.utils import secure_filename
import os
from recommender import FashionRecommender
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin ,LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash



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

# recommender = FashionRecommender('Fashion_Recommendation_System/fashion_products_sample.csv')
recommender = FashionRecommender('Fashion_Recommendation_System/fashion_data.pkl')

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


@app.route('/')
def home():
    return render_template('index.html')

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


@app.route('/recommend/name', methods=['POST'])
def recommend_name():
    product_name = request.form['product_name']

    filters = {
        'category': request.form.getlist('category'),
        'brand': request.form.getlist('brand'),
        'gender': request.form.getlist('gender'),
        'classi': request.form.getlist('class'),
        'colour': request.form.getlist('colour')
    }

    results, vector = recommender.recommend_by_name(product_name, filters=filters)
    if vector is not None:
        log_user_action('name', product_name, vector)
    return render_template('product.html', recommendations=results.to_dict(orient='records'))


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