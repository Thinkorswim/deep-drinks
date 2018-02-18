# Methods to manipulate text strings

# Imports
import re
import string
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# British English vocabulary
# gb_dict = enchant.Dict('en_GB')
# this slows down too much so these operations are commented

# Stemmer (Porter)
stemmer = PorterStemmer()


def convert_camel_case(my_string):
    """
    Convert camel case in a string into space-splitted.
    """
    return \
        re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', my_string)


def clean_string(my_string):
    """
    Clean a string by applying a series of transformations to it.
    Cleaning means, in order:
        - Convert potential camel case into space-splitted
        - Lower the case
        - Replace all potential punctuation with spaces
        - Replace all potential digits with spaces
        - For each token:
            * Decode, if possible, to utf8 (if not, ditch it)
            * Check spelling is en_GB, if not use first suggested word
              (disabled)
            * Stem token
    Return dictionary with original and cleaned tokens.
    NOTE: this does not include the synonyms collapse used in the AWS lambdas.
    """

    # Lower the case
    my_string = my_string.lower()
    my_string = remove_digits(my_string)
    my_string = remove_punctuation(my_string)
    my_string = remove_stopwords(my_string)
    my_string = stem(my_string)
    my_string = ' '.join(my_string.split())
    return my_string


def find_remove_substring(my_string, my_substring):
    """Find a substring in a string and remove it from string, lowering case"""

    return string.replace(my_string.lower(), my_substring.lower(), '')


def remove_digits(my_string):
    """Remove digits from string if there are"""
    return ''.join([i for i in my_string if not i.isdigit()])


def remove_stopwords(my_string):
    new_string = ''
    for item in my_string.split(' '):
        if item not in stopwords.words('english') and item not in ['amp']:
            new_string += item
            new_string += ' '
    return new_string.strip()


def stem(my_string):
    new_string = ''
    for item in my_string.split(' '):
        new_string += stemmer.stem(item)
        new_string += ' '
    return new_string.strip()


def remove_punctuation(my_string):
    """
    Return string stripped of punctuation (becomes a space).
    """
    for char in punctuation:
        if char in my_string:
            my_string = my_string.replace(char, ' ')
    return my_string


def remove_selected_punctuation(my_string, excluded_chars):
    """
    Return string stripped of selected punctuation signs (becomes a list).
    """
    for char in punctuation:
        if char in my_string and char not in excluded_chars:
            my_string = my_string.replace(char, ' ')
    return my_string
