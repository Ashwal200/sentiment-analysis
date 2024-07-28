from flask import Flask, request, jsonify, render_template # type: ignore
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json

app = Flask(__name__)

# Load dataset
df = pd.read_csv('sampled_data.csv', header=None, names=['text', 'label'], delimiter=',', quoting=3)
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Predefined text processing and classification tools
vectorizers = {
    'tfidf': TfidfVectorizer(max_features=5000),
    'count': CountVectorizer(max_features=5000)
    # Add Word2Vec if necessary
}

classifiers = {
    'random_forest': RandomForestClassifier(random_state=42),
    'adaboost': AdaBoostClassifier(random_state=42),
    'svm': SVC(kernel='linear', random_state=42),
    'gradient_boosting': GradientBoostingClassifier(random_state=42),
    'logistic_regression': LogisticRegression(random_state=42)
}

# Train models and store accuracy
accuracies = {}

for vec_name, vectorizer in vectorizers.items():
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    for clf_name, classifier in classifiers.items():
        classifier.fit(X_train_vect, y_train)
        y_pred = classifier.predict(X_test_vect)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[f'{vec_name}_{clf_name}'] = accuracy
        joblib.dump((vectorizer, classifier), f'{vec_name}_{clf_name}.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_accuracy', methods=['GET'])
def get_accuracy():
    vec_name = request.args.get('vectorizer')
    clf_name = request.args.get('classifier')
    accuracy = accuracies.get(f'{vec_name}_{clf_name}', 'Invalid combination')
    return jsonify({'accuracy': accuracy})

@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data)
    sentence = data['sentence']
    vec_name = data['vectorizer']
    clf_name = data['classifier']
    
    vectorizer, classifier = joblib.load(f'{vec_name}_{clf_name}.pkl')
    sentence_vect = vectorizer.transform([sentence])
    prediction = classifier.predict(sentence_vect)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
