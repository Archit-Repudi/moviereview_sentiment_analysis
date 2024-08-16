from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and the vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the form
    review = [request.form['review']]
    
    # Transform the review text to the TF-IDF vector
    transformed_review = vectorizer.transform(review)
    
    # Predict the sentiment
    prediction = model.predict(transformed_review)
    
    # Return the result
    result = 'Positive' if prediction[0] == 1 else 'Negative'
    
    return render_template('index.html', prediction_text=f'Sentiment: {result}')

if __name__ == "__main__":
    app.run(debug=True)
