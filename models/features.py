from nltk import pos_tag, word_tokenize
import string

# Function to calculate additional features
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
    