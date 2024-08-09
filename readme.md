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
- **Accuracy**: 89.5%

## Feature Engineering

To enhance our model's performance, we incorporated additional features that capture various aspects of the text data. These features were selected to provide the model with more context and improve its ability to predict sentiment accurately.
- **Word Count**: Counts the total number of words in a text.
- **Average Word Length**: Calculates the average length of words in a text.
- **Punctuation Count**: Counts the number of punctuation marks in a text.
- **Stopword Count**: Counts the number of common stopwords in a text (e.g., "the," "and").
- **Digit Count**: Counts the number of digits or numbers in a text.
- **POS Tagging (Noun Count)**: Counts the number of nouns in a text using Part-of-Speech tagging.
- **Exclamation Count**: Counts the number of exclamation marks, which may indicate strong emotions.
- **Question Count**: Counts the number of question marks, which may suggest inquisitiveness or doubt.
- **Title Word Count**: Counts the number of title-cased words, which might indicate proper nouns or emphasis.

## Training Techniques

1. **Pipeline**: Integrated preprocessing and model training steps to streamline the workflow.
2. **GridSearchCV**: Used for hyperparameter tuning to find the best model settings.
3. **Feature Integration**: Added custom features to improve model accuracy and performance.
1. **SelectKBest**: Focuses on the most significant features, reducing dimensionality and improving model results.

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
3. Feel free to explore our application...

## Conclusion

Our project successfully identified the most effective model and feature extraction method for sentiment analysis. Gradient Boosting with Count Vectorizer achieved the highest performance, demonstrating the importance of model selection and feature engineering. Future work will focus on further improving feature engineering and exploring advanced models.

## Future Work

- **Feature Engineering**: Continue experimenting with new features to enhance model performance.
- **Advanced Models**: Explore other advanced models and techniques.
- **Integration**: Consider integrating more complex text processing techniques.
