import numpy as np
import nltk
from nltk.stem import PorterStemmer
import jellyfish  # Library for Soundex and Levenshtein distance
# nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Split sentence into an array of words/tokens.
    A token can be a word, punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stemming = find the root form of the word.
    Examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def soundex_distance(word1, word2):
    """
    Calculate the Soundex distance between two words.
    """
    return jellyfish.soundex(word1) == jellyfish.soundex(word2)

def levenshtein_distance(word1, word2):
    """
    Calculate the Levenshtein (edit) distance between two words.
    """
    return jellyfish.levenshtein_distance(word1, word2)

def bag_of_words(tokenized_sentence, words):
    """
    Return a bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    Include handling for typos using Soundex and edit distance.
    """
    # Stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    
    for idx, w in enumerate(words):
        # Check for an exact match
        if w in sentence_words:
            bag[idx] = 1
        else:
            # Handle typos using Soundex and edit distance
            for word in sentence_words:
                if soundex_distance(w, word) and levenshtein_distance(w, word) <= 1:
                    bag[idx] = 1
                    break

    return bag
