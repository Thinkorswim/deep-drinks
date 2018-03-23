#!/usr/bin/env python3

# Disable warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Handle library imports
from helpers import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Embedding
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
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
def create_model_and_train(X, y, num_epochs, batch_size, sequence_len, embedding_dim, vocabulary_size, creativity, tokenizer):
    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=sequence_len,
                                trainable=False)
    model = Sequential()
    model.add(embedding_layer)
    model.add(GRU(256, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=vocabulary_size + 1, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    outputs = []
    for _ in range(num_epochs):
        result = model.fit(X, y, validation_split=0.05, shuffle=False, epochs=1, batch_size=batch_size, verbose=1)
        recipe_length = 200
        recipes = []
        for c in creativity:
            recipe = get_recipe(X, model, recipe_length, c, batch_size, tokenizer)
            recipes.append(recipe)

        output = {
            'history': result.history,
            'recipes': recipes,
            'sequence_len': sequence_len,
            'vocabulary_size': vocabulary_size,
            'embedding_dimension': embedding_dim
        }
        outputs.append(output)
        
    return outputs, model

###########################################
###########################################
###########################################

# Define useful helper functions
def get_recipe(X, model, recipe_length, creativity, batch_size, tokenizer):
    start = np.random.randint(0, len(X)-1)
    observation = X[start]
    seed = ' '.join(decode_label(observation, tokenizer))

    result, prediction = [], None

    for i in range(recipe_length):
        raw_prediction = model.predict(x=np.array([observation]), batch_size=batch_size)
        prediction = sample_distribution(raw_prediction, c=creativity)
        result.append(prediction)
        observation = np.append(observation[1:], prediction)

    output = ' '.join(decode_label(result, tokenizer))
    recipe = {
        'seed': seed,
        'output': output,
        'creativity': creativity
    }
    return recipe

def store_data(output, model, output_file):
    data = {'output': output, 'model': model.get_config()}
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    logger.info(">> Data saved to {}".format(output_file))
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
    parser.add_argument('-s', '--sequence', type=int)
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
        'data/liquor.json',
        'data/social_cocktail.json',
        'data/serious_eats.json',
        'data/live_in_style.json',
        'data/all_recipes.json'
    ]

    descriptions, ingredients, names = [], [], []

    for d in data:
        descriptions += load_data(d, field='description')
        ingredients += load_data(d, field='ingredients')
        names += load_data(d, field='name')

    assert len(descriptions) == len(ingredients)

    recipes = [x + ' # ' + y for x, y in zip(ingredients, descriptions)]
    logger.info('>> There are {} recipes in the database.'.format(len(recipes)))

    # Perform data preprocessing
    filter_words = ['a', 'an', 'the', 'fluid', '1', 'ounce', 'ounces']
    recipes = filter_texts(recipes, filter_words)

    # Set constant parameters used for all experiments
    sequence_len = args.sequence if args.sequence else 50
    embedding_dim = 300
    vocabulary_size = None
    glove_file = 'data/glove.42B.300d.txt'
    num_epochs = args.epochs if args.epochs else 10
    batch_size = 32

    # Transform input data
    tokenizer = Tokenizer(
        num_words=vocabulary_size,
        filters='!"$%&()*+,-:;<=>?@[\\]^_`{}~\t\n'
    )
    tokenizer.fit_on_texts(recipes)
    sequences = tokenizer.texts_to_sequences(recipes)
    word_index = tokenizer.word_index
    vocabulary_size = len(word_index) + 1 if vocabulary_size is None else vocabulary_size
    creativity = [1] + list(range(500, vocabulary_size, 500)) + [vocabulary_size]

    logger.info('>> Vocabulary size is {}.'.format(vocabulary_size))
    logger.info('>> Sequence length is {}.'.format(sequence_len))
    logger.info('>> creativity parameters are {}.'.format(creativity))
    logger.info('>> Number of epochs is {}.'.format(num_epochs))

    logger.info('>> Creating embedding matrix...')
    embedding_matrix = get_embedding_matrix(glove_file, word_index, vocabulary_size, embedding_dim)

    # Transform data into sequences and predictions
    X_recipes = make_flat(sequences)
    X, y = sequence_transform(X_recipes, sequence_len)
    logger.info('>> Observation shape is {}, label shape is {}'.format(X.shape, y.shape))

    output, model = create_model_and_train(X, y, num_epochs, batch_size, sequence_len, embedding_dim, vocabulary_size, creativity, tokenizer)
    store_data(output, model, output_file)
