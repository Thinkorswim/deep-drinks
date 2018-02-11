from string import punctuation
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import Counter
import numpy as np
import json

from IPython import embed

#############################
#############################
#############################

# Data loading helper functions
def load_data(data_file, field=None):
    data = []
    with open(data_file) as f:
        raw_data = json.load(f)
        if not field:
            return raw_data
        for obj in raw_data:
            data.append(obj[field])
    return data

# Can be used to return a list of words given a list of list of words
def flatten_list(deep_list):
    flatten_list = []
    for x in deep_list:
        flatten_list += x.split() + ['|']
    return flatten_list

# Limit the vocabulary of words to the most common N items
def limit_vocabulary(X, vocabulary_size=50, filter_words=None):
    if filter_words is None:
        filter_words = ['a', 'an', 'the']
    limit = vocabulary_size + len(filter_words)
    vocabulary = Counter(X).most_common(limit)
    vocabulary = set([x[0] for x in vocabulary if x[0] not in filter_words])
    X = [x for x in X if x in vocabulary]
    return X

# def sequence_transform(X, seq_length):
#     X_encoded, y_encoded = [], []
#     for i in range(len(X) - seq_length - 1):
#         if (i % 2500) == 0: print('>> DEBUG: Processed {} items'.format(i)) 
#         x = X[i:i+seq_length]
#         X_encoded.append(x.tolist())
#         y_encoded.append(X[i+seq_length+1].tolist())
#     return np.array(X_encoded), np.array(y_encoded)


# Breaks one data list into smaller training observations and labels
# for instance, X = [a, b, c, d, e] and seq_length = 2 yields
# X_encoded = [[a, b], [b, c], [c, d]
# y_encoded = [c, d, e]
def sequence_transform(X, n):
    X_encoded = [X[i:i+n] for i in range(len(X)-n+1)][:-1]
    y_encoded = X[n:]
    return np.array(X_encoded), np.array(y_encoded)

#############################
#############################
#############################

# Text manipulation functions
def clean_string(my_string):
    my_string = remove_punctuation(my_string)
    my_string = my_string.replace('.', ' .')
    my_string = my_string.lower()
    my_string = my_string.replace('  ', ' ')
    return my_string.strip()

def remove_punctuation(my_string):
    for char in punctuation:
        if char != '.' and char in my_string:
            my_string = my_string.replace(char, ' ')
    return my_string

#############################
#############################
#############################

# Neural network helper functions
# Helper Routines for Neural Networks

# Reverses one-hot encoding
def reverse_encoding(y_encoded, label_encoder, onehot_encoder):
    b = np.zeros_like(y_encoded)
    b[np.arange(len(y_encoded)), y_encoded.argmax(1)] = 1
    # Reverse One-Hot & Label Encoding
    decoded = b.dot(onehot_encoder.active_features_).astype(int)
    result = label_encoder.inverse_transform(decoded)
    return result.reshape(1, -1)[0]

# Produces one-hot encoding for categorical data
def encode_categorical(y, label_encoder=None, onehot_encoder=None):
    if label_encoder is None:
        label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y).reshape(-1, 1)
    if onehot_encoder is None:
        onehot_encoder = OneHotEncoder(sparse=False)
    y_encoded = onehot_encoder.fit_transform(y_encoded)
    assert np.array_equal(y, reverse_encoding(y_encoded, label_encoder, onehot_encoder))
    # y = y[:, 1:] # avoid dummy variable trap
    return y_encoded, label_encoder, onehot_encoder

# Predicts x (in one-hot encoding) given a model, and presents data
# as human-readable (raw_prediction = False), or as one-hot encoded (raw_prediction = True)
def predict_observation(model, x, batch_size, label_encoder, onehot_encoder, raw_prediction=False):
    prediction = model.predict(x=x, batch_size=batch_size)
    # Round into One-Hot Encoding
    b = np.zeros_like(prediction)
    b[np.arange(len(prediction)), prediction.argmax(1)] = 1
    if raw_prediction: return b
    # Reverse One-Hot & Label Encoding
    decoded = b.dot(onehot_encoder.active_features_).astype(int)
    result = label_encoder.inverse_transform(decoded)
    return result

def n_max(arr, n):
    return arr.flatten().argsort()[-n:][::-1]

# Predicts x (in one-hot encoding) given a model, and presents data
# as human-readable (raw_prediction = False), or as one-hot encoded (raw_prediction = True)
def predict_observation_with_rules(model, x, batch_size, label_encoder, onehot_encoder, last_prediction, raw_prediction=False):    
    prediction = model.predict(x=x, batch_size=batch_size)
    # Round into One-Hot Encoding
    top_2 = n_max(prediction, 2)
    a = np.zeros_like(prediction)
    b = np.zeros_like(prediction)
    c = np.zeros_like(prediction)
    a[np.arange(len(prediction)), prediction.argmax(1)] = 1
    b[np.arange(len(prediction)), top_2[0]] = 1
    c[np.arange(len(prediction)), top_2[1]] = 1
    assert np.array_equal(a, b)
    if last_prediction is not None and np.array_equal(b, last_prediction):
        print('WARN! trying to predict 2 words in a row')
        b = c
    if raw_prediction: return b
    # Reverse One-Hot & Label Encoding
    decoded = b.dot(onehot_encoder.active_features_).astype(int)
    result = label_encoder.inverse_transform(decoded)
    return result
