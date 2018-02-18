#!/usr/bin/env python3

# Disable warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Handle library imports
from char_level_helpers import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, GRU
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import numpy as np
import logging
import pickle
import argparse
import gc

###########################################
###########################################
###########################################

# Define functions for manipulating data and training models

def transform_data(data, seq_length, vocabulary_size):
    x = np.array([])

    for d in data:
        x = np.append(x,list(d)+['|'])

    X_limit_transformed, label_encoder, onehot_encoder = encode_categorical(x) # one-hot encode the data
    encoders = (label_encoder, onehot_encoder)
    # Transform data into sequences and predictions
    X, y = sequence_transform(X_limit_transformed, seq_length)
    logger.info('>> Observation shape is {}, label shape is {}'.format(X.shape, y.shape))
    return X, y, encoders

def grid_search(data, params, sequence_lengths, vocabulary_sizes, output_file):
    for seq in sequence_lengths:
        for vocab in vocabulary_sizes:
            logger.info(">>>> Starting experiment with sequence length: {} and vocabulary size: {}..."
                .format(seq, vocab))
            X, y, encoders = transform_data(data, seq, vocab)
            params['encoders'] = encoders
            output, model = create_model_and_train(X, y, params)
            store_data(output, model, seq, vocab, output_file)
            gc.collect() # use garbage collector
    return

def create_model_and_train(X, y, params):
    # Unpack parameters
    epochs = params['num_epochs']
    batch_size = params['batch_size']
    input_shape = (X.shape[1], X.shape[2])
    output_units = y.shape[1]
    model = params['model_fn'](params, input_shape, output_units)

    result = model.fit(X, y, validation_split=0.05, epochs=epochs, batch_size=batch_size, verbose=1)

    label_encoder, onehot_encoder = params['encoders']
    seq_length, vocabulary_size = input_shape
    recipe_length = 200
    recipe = get_recipe(X, model, recipe_length, seq_length, vocabulary_size, batch_size, label_encoder, onehot_encoder)

    output = {
        'history': result.history,
        'final_recipe': recipe,
        'sequence_length': seq_length,
        'vocabulary_size': vocabulary_size
    }

    return output, model

###########################################
###########################################
###########################################

# Define useful helper functions

def get_seed(X, seq_length, label_encoder, onehot_encoder):
    # Get seed in text form and seed encoded ('observation')
    start = np.random.randint(0, len(X)-1)
    seed = ''
    for i in range(seq_length):
        x = X[start, i].reshape(1, -1)
        seed += reverse_encoding(x, label_encoder, onehot_encoder)[0] + ' '
    seed = seed.strip()
    observation = X[start]
    return seed, observation

def get_recipe(X, model, recipe_length, seq_length, vocabulary_size, batch_size, label_encoder, onehot_encoder):
    seed, observation = get_seed(X, seq_length, label_encoder, onehot_encoder)
    result, prediction = [], None
    for i in range(recipe_length):
        prediction = predict_observation(
            model, np.array([observation]), batch_size, label_encoder, onehot_encoder, raw_prediction=True
        )
        result.append(prediction)
        observation = np.vstack((observation[1:, :], prediction))
    result = np.array(result).reshape(recipe_length, vocabulary_size)
    recipe = reverse_encoding(result, label_encoder, onehot_encoder)
    recipe = ' '.join(recipe)
    return recipe

def store_data(output, model, seq, vocab, output_file):
    filename = output_file + '_{}_{}.pickle'.format(seq, vocab)
    data = {'output': output, 'model': model.get_config()}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    logger.info(">> Data saved to {}".format(filename))
    return

###########################################
###########################################
###########################################

# Main function
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a trainer for a Deep Neural Network.')
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-e', '--epochs', type=int)
    args = parser.parse_args()

    output_file = 'output/{}'.format(args.output)

    # Seed a random number generator
    seed = 10102016
    rng = np.random.RandomState(seed)

    # Set up a logger object to print info about the training run to stdout
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = [logging.StreamHandler()]

    # Load training data
    data = [
        'data/social_cocktail.json',
        'data/liquor.json'
    ]
    descriptions = []
    for d in data:
        descriptions += load_data(d, field='description')
    logger.info('>>>> There are {} recipes in the database.'.format(len(descriptions)))

    # Perform data preprocessing
    descriptions = [clean_string(x) for x in descriptions]
    data = np.array(descriptions)

    # Set constant parameters used for all experiments
    num_epochs = args.epochs if args.epochs else 10
    batch_size = 100

    # Define Grid Search hyperparameters
    sequence_lengths = [10, 25, 50] # [5, 10, 15]
    vocabulary_sizes = [63] # [300, 500, 700]

    # Define neural network
    def create_model(params, input_shape, output_units):
        model = Sequential()
        model.add(GRU(256, input_shape=input_shape, return_sequences=True, activation='relu'))
        model.add(Dropout(0.2))
        model.add(GRU(128, input_shape=input_shape, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(output_units, activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
        return model

    params = {
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'model_fn': create_model
    }

    grid_search(data=data, params=params, sequence_lengths=sequence_lengths, vocabulary_sizes=vocabulary_sizes, output_file=output_file)
