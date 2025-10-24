from flask import Flask, request, render_template
import pandas as pd
import random
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle # New import for saving/loading ML models
from dotenv import load_dotenv # New import for environment variables

# --- 1. ENVIRONMENT/SECURITY SETUP (NEW) ---
# Load environment variables from .env file
load_dotenv() 
# Use a .env file to store the sensitive URI 
# Example .env content: DATABASE_URI="mysql+pymysql://root:@localhost/ecom"

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'models')


# load files===========================================================================================================
trending_products = pd.read_csv(os.path.join(DATA_PATH, "trending_products.csv"))
train_data = pd.read_csv(os.path.join(DATA_PATH, "clean_data.csv"))


# --- 2. PRE-CALCULATE ML COMPONENTS (NEW) ---

# Define paths for saved models
TFIDF_PATH = os.path.join(DATA_PATH, 'tfidf_matrix.pkl')
SIMILARITY_PATH = os.path.join(DATA_PATH, 'cosine_similarity.pkl')

tfidf_matrix_content = None
cosine_similarities_content = None

try:
    # Attempt to load pre-calculated data
    with open(TFIDF_PATH, 'rb') as f:
        tfidf_matrix_content = pickle.load(f)
    with open(SIMILARITY_PATH, 'rb') as f:
        cosine_similarities_content = pickle.load(f)
    print("Pre-calculated ML models loaded successfully.")

except FileNotFoundError:
    # If not found, calculate them once at startup
    print("Pre-calculated ML models not found. Calculating now...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    
    # Save the calculated models for faster startup next time
    with open(TFIDF_PATH, 'wb') as f:
        pickle.dump(tfidf_matrix_content, f)
    with open(SIMILARITY_PATH, 'wb') as f:
        pickle.dump(cosine_similarities_content, f)
    print("ML models calculated and saved.")


# database configuration---------------------------------------
app.secret_key = os.getenv('SECRET_KEY', "alskdjfwoeieiurlskdjfslkdjf") # Better to use os.getenv
# Retrieve the database URI from the environment variables (or use hardcoded as fallback)
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URI', "mysql+pymysql://root:@localhost/ecom")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Recommendations functions============================================================================================
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


def content_based_recommendations(train_data, item_name, top_n=10):
    global cosine_similarities_content # Use the global pre-calculated variable

    # Check if the item name exists in the training data
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    # NOTE: The TFIDF calculation is removed from here! It's pre-calculated above.

    # Find the index of the item
    # Use .iloc[0] for safety in case of multiple matches
    try:
        item_index = train_data[train_data['Name'] == item_name].index.to_list()[0]
    except IndexError:
        print(f"Error finding index for {item_name}")
        return pd.DataFrame()

    # Get the cosine similarity scores for the item (using pre-calculated matrix)
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n+1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details
# routes===============================================================================
# List of predefined image URLs
random_image_urls = [
    "static/img/img_1.png",
    "static/img/img_2.png",
    "static/img/img_3.png",
    "static/img/img_4.png",
    "static/img/img_5.png",
    "static/img/img_6.png",
    "static/img/img_7.png",
    "static/img/img_8.png",
]


@app.route("/")
def index():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=trending_products.head(8),truncate = truncate,
                            random_product_image_urls=random_product_image_urls,
                            random_price = random.choice(price))

# --- FIX: Main page needs to pass an empty variable (or modify HTML) ---
@app.route("/main")
def main():
    # Render template, passing an empty value to satisfy main.html's expected variable.
    # This prevents the UndefinedError when the page is loaded directly.
    return render_template('main.html', content_based_rec=pd.DataFrame()) 

# routes
@app.route("/index")
def indexredirect():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                            random_product_image_urls=random_product_image_urls,
                            random_price=random.choice(price))

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                                random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                                signup_message='User signed up successfully!'
                                )

# Route for signup page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        new_signup = Signin(username=username,password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                                random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                                signup_message='User signed in successfully!'
                                )
@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')
        # Safely convert to integer
        try:
            nbr = int(request.form.get('nbr'))
        except (ValueError, TypeError):
            nbr = 10 # Default value

        content_based_rec = content_based_recommendations(train_data, prod, top_n=nbr)

        if content_based_rec.empty:
            message = f"No recommendations available for product '{prod}'."
            return render_template('main.html', message=message, content_based_rec=pd.DataFrame()) # Pass empty DF to prevent crash
        else:
            # Create a list of random image URLs for each recommended product
            # NOTE: This uses the length of trending_products, which may not match recommended products.
            # It should ideally use len(content_based_rec)
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(content_based_rec))]
            
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=content_based_rec, truncate=truncate,
                                    random_product_image_urls=random_product_image_urls,
                                    random_price=random.choice(price))


if __name__=='__main__':
    # Initialize the database within the application context (necessary for flask-sqlalchemy)
    # The database must be running for this to work!
    with app.app_context():
        # db.create_all() # Uncomment this line if you need to create your database tables
        pass
        
    app.run(debug=True)