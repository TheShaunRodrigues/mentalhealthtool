from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from textblob import TextBlob
import sqlite3
from cryptography.fernet import Fernet

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('phq-9.csv')

# Calculate PHQ-9 score by summing up columns phq1 to phq9
data['phq9_total'] = data[['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']].sum(axis=1)

# Function to interpret PHQ-9 score and categorize severity
def interpret_phq9_score(score):
    if score <= 4:
        return "Minimal or no depression"
    elif 5 <= score <= 9:
        return "Mild depression"
    elif 10 <= score <= 14:
        return "Moderate depression"
    elif 15 <= score <= 19:
        return "Moderately severe depression"
    else:
        return "Severe depression"

# Add the calculated severity level to the dataset
data['depression_severity'] = data['phq9_total'].apply(interpret_phq9_score)

# Train the Machine Learning model (RandomForestClassifier)
X = data[['phq1', 'phq2', 'phq3', 'phq4', 'phq5', 'phq6', 'phq7', 'phq8', 'phq9']]  # Features (PHQ-9 questions)
y = data['depression_severity']  # Target (calculated severity)
model = RandomForestClassifier()
model.fit(X, y)

# Function for sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Encrypt and Decrypt Data (Secure storage)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_data(data):
    return cipher_suite.encrypt(data.encode())

def decrypt_data(data):
    return cipher_suite.decrypt(data).decode()

# Save assessments to SQLite
def save_assessment(user_id, responses, score, sentiment):
    conn = sqlite3.connect('phq9_assessments.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS assessments
                      (user_id TEXT, responses TEXT, score INTEGER, sentiment REAL)''')
    cursor.execute("INSERT INTO assessments VALUES (?, ?, ?, ?)",
                   (user_id, str(responses), score, sentiment))
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def phq9_assessment():
    if request.method == 'POST':
        responses = [int(request.form[f'q{i+1}']) for i in range(9)]
        text_input = request.form['mood']
        
        # Calculate PHQ-9 score
        score = sum(responses)
        interpretation = interpret_phq9_score(score)
        
        # Analyze sentiment
        sentiment = analyze_sentiment(text_input)
        sentiment_feedback = f"Your sentiment score is {sentiment:.2f}, indicating a {'positive' if sentiment > 0 else 'negative'} mood."
        
        # Save assessment (Optionally encrypt data)
        user_id = "user123"  # Example user ID, replace with actual user data
        save_assessment(user_id, responses, score, sentiment)
        
        return render_template('result.html', total_score=score, mood_description=text_input, interpretation=interpretation, sentiment=sentiment_feedback)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
