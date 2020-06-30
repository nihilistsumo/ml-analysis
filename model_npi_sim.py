import json
import math
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras import regularizers
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Activation, Input
from tensorflow.keras.utils import to_categorical
import argparse
import csv
import sys


import random
random.seed(42)

def sample_npi_groups(groups_in_split, group_npi_mapping):
    npis_in_split = []
    for group in groups_in_split:
        npis_in_split += group_npi_mapping[group]
    labelled_npi_pairs = {}
    for group in groups_in_split:
        npis_in_group = group_npi_mapping[group]
        npis_not_in_group = [n for n in npis_in_split if n not in npis_in_group]
        positive_samples = 0
        negative_samples = 0
        for i in range(len(npis_in_group)-1):
            for j in range(i+1, len(npis_in_group)):
                npi_key = [npis_in_group[i], npis_in_group[j]]
                npi_key.sort()
                labelled_npi_pairs[tuple(npi_key)] = 1
                positive_samples += 1
        if positive_samples > 0:
            npi1_index = 0
            while(negative_samples < positive_samples):
                npi1 = npis_in_group[npi1_index % len(npis_in_group)]
                if len(npis_not_in_group) < 1:
                    npis_not_in_group = [n for n in npis_in_split if n not in npis_in_group]
                npi2 = random.sample(npis_not_in_group, 1)[0]
                npis_not_in_group.remove(npi2)
                npi_key = [npi1, npi2]
                npi_key.sort()
                labelled_npi_pairs[tuple(npi_key)] = 0
                negative_samples += 1
                npi1_index += 1
    return labelled_npi_pairs

def get_data(npi_embed_file_path, npi_group_file_path):
    with open(npi_embed_file_path, 'r') as npi_embed_file:
        npi_embed_data = json.load(npi_embed_file)
    npi_groups = {}
    with open(npi_group_file_path, 'r') as npi_group_file:
        file_reader = csv.reader(npi_group_file)
        next(file_reader)
        for row in file_reader:
            group = row[0]
            npi = row[1]
            if group not in npi_groups.keys():
                npi_groups[group] = [npi]
            else:
                npi_groups[group].append(npi)
    return (npi_embed_data, npi_groups)

def get_npi_data_matrix(npi_pairs_data, npi_embed_data):
    X = []
    y = []
    for pair in npi_pairs_data.keys():
        npi1, npi2 = pair
        npi1data = npi_embed_data[npi1]
        npi2data = npi_embed_data[npi2]
        npi1_taxonomy_encoding = []
        npi2_taxonomy_encoding = []
        for code in npi1data['taxonomy']:
            npi1_taxonomy_encoding += code
        for code in npi2data['taxonomy']:
            npi2_taxonomy_encoding += code
        npi1vec = npi1data['name'] + npi1data['other'] + npi1data['location'] + npi1_taxonomy_encoding
        npi2vec = npi2data['name'] + npi2data['other'] + npi2data['location'] + npi2_taxonomy_encoding
        X.append(npi1vec + npi2vec)
        y.append(npi_pairs_data[pair])
    return X, y

def prepare_dataset(npi_embed_data, train_npi_pairs, test_npi_pairs):
    (X_train, y_train) = get_npi_data_matrix(train_npi_pairs, npi_embed_data)
    (X_test, y_test) = get_npi_data_matrix(test_npi_pairs, npi_embed_data)

    X_train = np.array(X_train)
    y_train = to_categorical(np.array(y_train))
    X_test = np.array(X_test)
    y_test = to_categorical(np.array(y_test))
    return (X_train, y_train), (X_test, y_test)

def run_logistic_regression(X_train, y_train, X_test=None, y_test=None):
    veclen = X_train.shape[1]
    model = Sequential()
    model.add(Input(shape=(veclen,), dtype='float32', name='vec1'))
    model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.L1L2(l1=0.0, l2=0.1)))
    batch_size = 128
    num_epoch = 100
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    if X_test is not None:
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1,
                        validation_data=(X_test, y_test))
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return score
    else:
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1)
        return model

def run_neural_network(X_train, y_train, hidden_layers, X_test=None, y_test=None):
    veclen = X_train.shape[1]
    model = Sequential()
    model.add(Input(shape=(veclen,), dtype='float32', name='vec1'))
    for layer in hidden_layers:
        model.add(Dense(layer, activation='relu', kernel_regularizer=regularizers.L1L2(l1=0.0, l2=0.01)))
    model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.L1L2(l1=0.0, l2=0.01)))
    batch_size = 128
    num_epoch = 100
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    if X_test is not None:
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1,
                        validation_data=(X_test, y_test))
        score = model.evaluate(X_test, y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        return score
    else:
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, verbose=1)
        return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', '--npi_embed_file', help='Path to npi embedding file generated by npi_embedding.py in '
                                                        'json format')
    parser.add_argument('-ng', '--npi_groups_file', help='Path to npi groups mapping file')
    parser.add_argument('-m', '--model', type=int, help='Choice of model: 1 - Logistic regression, 2 - neural')
    parser.add_argument('-hl', '--hidden_layers', type=int, nargs='+', help='List of hidden layers sizes')
    parser.add_argument('-op', '--options', type=int, help='Options: 1 - Evaluate the model using 5 fold CV, 2 - Train'
                                                           'the model using the full dataset')
    parser.add_argument('-mo', '--model_output', default='model.out', help='Path to save the trained model '
                                                      '(used only if 2  is chosen for option)')
    args = parser.parse_args()
    npi_embed_file_path = args.npi_embed_file
    npi_groups_file_path = args.npi_groups_file
    model = args.model
    option = args.options

    (npi_embed_data, npi_group_map) = get_data(npi_embed_file_path, npi_groups_file_path)

    if option == 1:
        group_indices = list(npi_group_map.keys())
        group_partitions = [group_indices[i::5] for i in range(5)]
        f = 1
        scores = []
        for test_groups in group_partitions:
            print("Fold "+str(f)+"\n+++++++++++++")
            train_groups = [k for k in group_indices if k not in test_groups]
            train_npi_pairs = sample_npi_groups(train_groups, npi_group_map)
            test_npi_pairs = sample_npi_groups(test_groups, npi_group_map)
            (X_train, y_train), (X_test, y_test) = prepare_dataset(npi_embed_data, train_npi_pairs, test_npi_pairs)
            f += 1
            if model == 1:
                scores.append(run_logistic_regression(X_train, y_train, X_test, y_test))
            else:
                hidden_layers = args.hidden_layers
                if hidden_layers is None or len(hidden_layers) < 1:
                    print('Corrupted hidden layer info')
                    sys.exit(0)
                scores.append(run_neural_network(X_train, y_train, hidden_layers, X_test, y_test))
        scores = np.array(scores)
        for i in range(len(scores)):
            print('Fold '+str(i)+' score: %0.4f, accuracy: %0.4f' % (scores[i, 0], scores[i, 1]))
        print('Mean score: %0.4f' % scores[:, 0].mean())
        print('Mean accuracy: %0.4f' % scores[:, 1].mean())
    elif option == 2:
        model_out = args.model_output
        train_npi_pairs = sample_npi_groups(list(npi_group_map.keys()), npi_group_map)
        (X_train, y_train) = get_npi_data_matrix(train_npi_pairs, npi_embed_data)
        X_train = np.array(X_train)
        y_train = to_categorical(np.array(y_train))
        if model == 1:
            m = run_logistic_regression(X_train, y_train)
        else:
            hidden_layers = args.hidden_layers
            if hidden_layers is None or len(hidden_layers) < 1:
                print('Corrupted hidden layer info')
                sys.exit(0)
            m = run_neural_network(X_train, y_train, hidden_layers)
        m.save(model_out)
    else:
        print('Wrong option')

if __name__ == '__main__':
    main()