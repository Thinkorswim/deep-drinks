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

# Can be used to return a list of items given a list of list of items
def make_flat(deep_list):
    flat_list = []
    for x in deep_list:
        flat_list += x
    return flat_list

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

# Breaks one data list into smaller training observations and labels
# for instance, X = [a, b, c, d, e] and seq_length = 2 yields
# X_encoded = [[a, b], [b, c], [c, d]
# y_encoded = [c, d, e]
def sequence_transform(X, n):
    X_encoded = [X[i:i+n] for i in range(len(X)-n+1)][:-1]
    y_encoded = X[n:]
    return np.array(X_encoded), np.array(y_encoded)

def get_embedding_matrix(glove_file, word_index, vocab_size, embedding_dim):
    embeddings_index = {}
    with open(glove_file, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

#############################
#############################
#############################

def filter_texts(texts, filter_words):
    remove_unicode = lambda x: x.encode('ascii', errors='ignore').decode().strip()
    texts = [remove_unicode(x) for x in texts]
    texts = [' '.join([y for y in x.split() if y not in filter_words]) for x in texts]
    texts = [x.replace('.', ' . ') for x in texts]
    texts = [x + ' |' for x in texts]
    texts = [x.replace('  ', ' ') for x in texts]
    np.random.shuffle(texts)
    return texts

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

# Reverse label encoding produced by tokenizer
def decode_label(encoded_sequence, tokenizer):
    decoded_sequence = []
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    for x in encoded_sequence:
        decoded_sequence.append(reverse_word_map[x])
    return decoded_sequence

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

def sample_distribution(distribution, c=1):
    D = distribution.shape[1]
    assert c <= D
    top_c = n_max(distribution, c)
    dist_c = distribution[:, top_c]
    dist_c = dist_c / dist_c.sum(axis=1,keepdims=1)
    i = np.random.choice(top_c, p=dist_c.flatten())
    return i

# Predicts x (in one-hot encoding) given a model, and presents data
# as human-readable (raw_prediction = False), or as one-hot encoded (raw_prediction = True)
def language_model_sampling(model, x, batch_size, label_encoder, onehot_encoder, last_prediction, raw_prediction=False, c=1):
    prediction = model.predict(x=x, batch_size=batch_size)
    # Round into One-Hot Encoding
    rand_idx = sample_distribution(prediction, c=c)
    b = np.zeros_like(prediction)
    b[np.arange(len(prediction)), rand_idx] = 1
    if raw_prediction: return b
    # Reverse One-Hot & Label Encoding
    decoded = b.dot(onehot_encoder.active_features_).astype(int)
    result = label_encoder.inverse_transform(decoded)
    return result

# Predicts x (in one-hot encoding) given a model, and presents data
# as human-readable (raw_prediction = False), or as one-hot encoded (raw_prediction = True)
def language_model_sampling_2(model, x, batch_size, raw_prediction=False, c=1):
    prediction = model.predict(x=x, batch_size=batch_size)
    # Round into One-Hot Encoding
    rand_idx = sample_distribution(prediction, c=c)
    b = np.zeros_like(prediction)
    b[np.arange(len(prediction)), rand_idx] = 1
    if raw_prediction:
        return b
    result = np.argmax(prediction, axis=None, out=None)
    return result
