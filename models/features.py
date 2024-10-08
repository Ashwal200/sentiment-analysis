from nltk import pos_tag, word_tokenize
import string
from nltk.corpus import stopwords
import nltk

# Ensure required NLTK data is downloaded
nltk.download('stopwords')

# Load stopwords
stop_words = set(stopwords.words('english'))

def calculate_text_features_gbc_tf(text):
    words = text.split()
    # Punctuation Count
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    # Digit Count
    digit_count = sum(1 for char in text if char.isdigit())
    # POS Tagging
    pos_tags = pos_tag(word_tokenize(text))
    noun_count = sum(1 for word, tag in pos_tags if tag.startswith('NN'))
    # Exclamation Count
    exclamation_count = text.count('!')
    # Question Count
    question_count = text.count('?')
    # Title Word Count
    title_word_count = sum(1 for word in words if word.istitle())
    
    return (punctuation_count, digit_count, noun_count,
            exclamation_count, question_count, title_word_count)

def calculate_text_features_gbc_cv(text):
    words = text.split()
    # Punctuation Count
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    # Digit Count
    digit_count = sum(1 for char in text if char.isdigit())
    # POS Tagging
    pos_tags = pos_tag(word_tokenize(text))
    noun_count = sum(1 for word, tag in pos_tags if tag.startswith('NN'))
    # Exclamation Count
    exclamation_count = text.count('!')
    # Question Count
    question_count = text.count('?')
    # Title Word Count
    title_word_count = sum(1 for word in words if word.istitle())
    
    return (punctuation_count, digit_count, noun_count,
            exclamation_count, question_count, title_word_count)

def calculate_text_features_svm_tf(text):
    words = text.split()
    # Punctuation Count
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    # Digit Count
    digit_count = sum(1 for char in text if char.isdigit())
    # POS Tagging
    pos_tags = pos_tag(word_tokenize(text))
    noun_count = sum(1 for word, tag in pos_tags if tag.startswith('NN'))
    # Exclamation Count
    exclamation_count = text.count('!')
    # Question Count
    question_count = text.count('?')
    # Title Word Count
    title_word_count = sum(1 for word in words if word.istitle())
    
    return (punctuation_count, digit_count, noun_count,
            exclamation_count, question_count, title_word_count)
    
def calculate_text_features_lr_cv(text):
    # Word Count
    word_count = len(text.split())
    # Average Word Length
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    # Punctuation Count
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    # Stopword Count
    stopword_count = sum(1 for word in words if word.lower() in stop_words)
    # Digit Count
    digit_count = sum(1 for char in text if char.isdigit())
    # POS Tagging
    pos_tags = pos_tag(word_tokenize(text))
    noun_count = sum(1 for word, tag in pos_tags if tag.startswith('NN'))
    # Exclamation Count
    exclamation_count = text.count('!')
    # Question Count
    question_count = text.count('?')
    # Title Word Count
    title_word_count = sum(1 for word in words if word.istitle())
    
    return (word_count, avg_word_length, punctuation_count, stopword_count, digit_count, noun_count,
            exclamation_count, question_count, title_word_count)

def calculate_text_features_lr_tf(text):
    # Word Count
    word_count = len(text.split())
    # Average Word Length
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    # Punctuation Count
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    # Stopword Count
    stopword_count = sum(1 for word in words if word.lower() in stop_words)
    # Digit Count
    digit_count = sum(1 for char in text if char.isdigit())
    # POS Tagging
    pos_tags = pos_tag(word_tokenize(text))
    noun_count = sum(1 for word, tag in pos_tags if tag.startswith('NN'))
    # Exclamation Count
    exclamation_count = text.count('!')
    # Question Count
    question_count = text.count('?')
    # Title Word Count
    title_word_count = sum(1 for word in words if word.istitle())
    
    return (word_count, avg_word_length, punctuation_count, stopword_count, digit_count, noun_count,
            exclamation_count, question_count, title_word_count)


    