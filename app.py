from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.over_sampling import SMOTE
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import json
import importlib.util
import numpy as np
import pandas as pd
from scipy.sparse import hstack

# import features
from models.features import calculate_text_features_gbc_tf as GBC_TF
from models.features import calculate_text_features_gbc_cv as GBC_CV
from models.features import calculate_text_features_svm_tf as SVM_TF
from models.features import calculate_text_features_lr_cv as LR_CV
from models.features import calculate_text_features_lr_tf as LR_TF
# ...

module_name_to_func = {
    'GBC_TF': GBC_TF,
    'GBC_CV': GBC_CV,
    'SVM_TF': SVM_TF,
    'LR_CV': LR_CV,
    'LR_TF': LR_TF,
}

# Custom tokenizer
stop_words = set(stopwords.words('english'))

def custom_tokenizer(text, stop_words=stop_words):
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token.isalnum() and token not in stop_words]

def tokenizer(text):
    return custom_tokenizer(text)

# Custom feature extractors
class TextLengthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([len(text.split()) for text in X]).reshape(-1, 1)

class ExclamationCountExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([text.count('!') for text in X]).reshape(-1, 1)

class UppercaseRatioExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0 for text in X]).reshape(-1, 1)

class SentimentScoreExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([sia.polarity_scores(text)['compound'] for text in X]).reshape(-1, 1)

class TextBlobFeaturesExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            blob = TextBlob(text)
            features.append([
                blob.sentiment.polarity,
                blob.sentiment.subjectivity,
            ])
        return np.array(features)

# Custom transformer for SMOTE
class SMOTETransformer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.smote = SMOTE(random_state=self.random_state)

    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        if y is not None:
            X_resampled, y_resampled = self.smote.fit_resample(X, y)
            return X_resampled
        return X

app = Flask(__name__)

# Load configuration from file
with open('config.json', 'r') as f:
    config = json.load(f)

model_paths = config['model_paths']
accuracies = config['accuracies']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_accuracy', methods=['GET'])
def get_accuracy():
    vectorizer = request.args.get('vectorizer')
    classifier = request.args.get('classifier')
    model_key = f"{vectorizer}_{classifier}"
    accuracy = accuracies.get(model_key, 'Invalid combination')
    return jsonify({'accuracy': accuracy})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence = data['sentence']
    vectorizer = data['vectorizer']
    classifier = data['classifier']
    model_key = f"{vectorizer}_{classifier}"
    
    if model_key in model_paths:
        vectorizer, classifier = joblib.load("trained_models/" + model_paths[model_key])
        sentence_vect = vectorizer.transform([sentence])
        
        # Load the correct feature calculation function
        try:
            calculate_text_features = module_name_to_func[model_paths[model_key][:-4]]
            additional_features = calculate_text_features(sentence)

            # Combine vectorized sentence with additional features
            sentence_combined_vect = hstack([sentence_vect, np.array([additional_features])])
        except:
            sentence_combined_vect = sentence_vect
            
        prediction = classifier.predict(sentence_combined_vect)
        print(prediction)
        return jsonify({'prediction': prediction[0]})
    
    else:
        return jsonify({'error': 'Invalid vectorizer or classifier'})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
