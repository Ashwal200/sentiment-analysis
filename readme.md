# Sentiment Analysis Project

## Overview

This project aims to classify text into six emotion labels using various machine learning models and feature extraction techniques. The dataset used is from Hugging Face, with the primary goal of identifying the best model and feature extraction method for sentiment analysis.

## Dataset

- **Source**: Hugging Face Dataset
- **Labels**: 6 emotions (e.g., anger, sadness, love, etc.)

## Process Tools

- **TF-IDF**: Term Frequency-Inverse Document Frequency
- **Count Vectorizer**: Converts text to a matrix of token counts

## Classifier Tools

1. **Gradient Boosting**: An ensemble method that builds multiple weak learners sequentially to improve accuracy.
2. **SVM (Support Vector Machine)**: Finds the optimal hyperplane that separates classes with the maximum margin.
3. **Logistic Regression**: Models class probabilities using the logistic function.
4. **Random Forest**: Combines multiple decision trees to improve predictive performance.
5. **Naive Bayes**: A probabilistic classifier based on Bayes' theorem with an assumption of feature independence.

## Best Results

- **Model**: Gradient Boosting
- **Feature Extraction**: Count Vectorizer
- **Accuracy**: 89.2%

## Feature Engineering

We added the following features to enhance our model's performance:
- **Word Count**
- **Average Word Length**
- **Punctuation Count**
- **Stopword Count**
- **Digit Count**
- **POS Tagging (Noun Count)**
- **Exclamation Count**
- **Question Count**
- **Title Word Count**

## Training Techniques

1. **Pipeline**: Integrated preprocessing and model training steps to streamline the workflow.
2. **GridSearchCV**: Used for hyperparameter tuning to find the best model settings.
3. **Feature Integration**: Added custom features to improve model accuracy and performance.

## Demo

Explore our app to:
- Choose your text processing tool (TF-IDF or Count Vectorizer).
- Select a classifier (Gradient Boosting, SVM, Logistic Regression, Random Forest, Naive Bayes).
- View the accuracy of the chosen model.
- Predict the emotion of a given sentence.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ashwal200/sentiment-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd sentiment-analysis
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://localhost:8000`.

## Conclusion

Our project successfully identified the most effective model and feature extraction method for sentiment analysis. Gradient Boosting with Count Vectorizer achieved the highest performance, demonstrating the importance of model selection and feature engineering. Future work will focus on further improving feature engineering and exploring advanced models.

## Future Work

- **Feature Engineering**: Continue experimenting with new features to enhance model performance.
- **Advanced Models**: Explore other advanced models and techniques.
- **Integration**: Consider integrating more complex text processing techniques.