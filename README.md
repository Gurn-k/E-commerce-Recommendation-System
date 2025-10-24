🛒 E-commerce Product Recommendation System

This project is a web-based e-commerce recommendation engine built in Python. It suggests relevant products to users based on features of items they have viewed or liked. The application uses Content-Based Filtering to analyze product descriptions and generate recommendations, displayed in a Flask web interface.

⚙️ Technologies Used

| Category             | Technology / Library | Purpose                                  |
| -------------------- | -------------------- | ---------------------------------------- |
| **Language**         | Python 3.x           | Core development                         |
| **Web Framework**    | Flask                | Serves the web application               |
| **Machine Learning** | Scikit-learn         | TF-IDF Vectorization & Cosine Similarity |
| **Data Handling**    | Pandas, NumPy        | Data cleaning & processing               |

🧠 Model Details

The system uses Content-Based Filtering by computing similarity between product descriptions.
Mechanism: TF-IDF Vectorization + Cosine Similarity
Model Output: cosine_similarity.pkl
Size: ~127 MB → Excluded from GitHub due to size limits (tracked in .gitignore)
Important: The web app will not work until this similarity matrix is generated locally.

💻 Local Setup & Installation

Follow these steps to run the project on your system:
1. Clone the Repository
git clone https://github.com/Gurn-k/E-commerce-Recommendation-System.git
cd E-commerce-Recommendation-System

2. Create & Activate Virtual Environment
# Create environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

▶️ Running the Application
Step 1: Generate the Model File
Open the notebook recommendations_code.ipynb
Run all cells to generate:
models/cosine_similarity.pkl

Step 2: Launch the Flask Server
python app.py

Step 3: Open in Browser
Visit:
http://127.0.0.1:5000/

📌 Notes
Ensure product data CSV and model file paths are correct before running.
Re-generating the similarity matrix may take time depending on dataset size.

🌟 Future Enhancements
Add collaborative filtering for personalized recommendations.
Integrate a database instead of static CSVs.
Deploy on cloud (Render / Railway / AWS / Azure).
