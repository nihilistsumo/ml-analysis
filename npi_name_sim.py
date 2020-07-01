import re
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.layers import Input, Embedding, LSTM, concatenate, Lambda
from tensorflow.python.keras.models import Model
import tensorflow.python.keras.backend as K
import tensorflow as tf

def text_to_word_list(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text

def build_name_corpus(npidata_file, output_file):
    npidata_dataframe = pd.read_csv(npidata_file)
    npi_names = list(npidata_dataframe['Provider Organization Name (Legal Business Name)'])
    npi_other_names = list(npidata_dataframe['Provider Other Organization Name'])
    name_corpus = []
    print(str(len(npi_names)) + ' records to process')
    processed_records = 0
    for i in range(len(npi_names)):
        name = npi_names[i]
        other_name = npi_other_names[i]
        processed_name = []
        if isinstance(name, str):
            processed_name += text_to_word_list(name)
        if isinstance(other_name, str):
            processed_name += text_to_word_list(other_name)
        if len(processed_name) > 0:
            name_corpus.append(processed_name)
        processed_records += 1
        if processed_records % 10000 == 0:
            print(str(processed_records) + ' records processed')
    with open(output_file, 'w') as out:
        for name_field in name_corpus:
            tokens = ''
            for token in name_field:
                tokens += token + ' '
            out.write(tokens.rstrip()+'\n')

def build_embeddings(glove_vec_file):
    glove_dataframe = pd.read_csv(glove_vec_file, sep=' ', quoting=3, header=None, index_col=0)
    glove_dict = {word: vecs.values for word, vecs in glove_dataframe.T.items()}
    vocabulary = list(glove_dict.keys())
    veclen = glove_dict[vocabulary[0]].size
    embedding_matrix = np.random.randn(len(vocabulary)+1, veclen)
    embedding_matrix[0] = 0
    for i in range(len(vocabulary)):
        embedding_matrix[i+1] = glove_dict[vocabulary[i]]
    del glove_dict
    return vocabulary, embedding_matrix

def get_npi_name_mappings(npidata_file, vocabulary):
    npidata_dataframe = pd.read_csv(npidata_file)
    npi_name_data = {}
    for index, row in npidata_dataframe.iterrows():
        npi = row['NPI']
        npi_name = row['Provider Organization Name (Legal Business Name)']
        npi_other_name = row['Provider Other Organization Name']
        npi_name_words = text_to_word_list(npi_name) if isinstance(npi_name, str) else []
        npi_other_name_words = text_to_word_list(npi_other_name) if isinstance(npi_other_name, str) else []
        npi_name_data[npi] = \
            {'name': [vocabulary.index(word) + 1 if word in vocabulary else 0 for word in npi_name_words],
             'other': [vocabulary.index(word) + 1 if word in vocabulary else 0 for word in npi_other_name_words]}
    max_name_len = max([len(npi_name_data[npi]['name']) for npi in npi_name_data.keys()])
    max_other_name_len = max([len(npi_name_data[npi]['other']) for npi in npi_name_data.keys()])
    for npi in npi_name_data.keys():
        npi_name_data[npi]['name'] = [0] * (max_name_len - len(npi_name_data[npi]['name'])) + npi_name_data[npi]['name']
        npi_name_data[npi]['other'] = [0] * (max_other_name_len - len(npi_name_data[npi]['other'])) + \
                                      npi_name_data[npi]['other']
    return npi_name_data, max_name_len, max_other_name_len

def get_npipairs_data_matrix(npi_pairs_data, npi_name_data):
    X_left = []
    X_right = []
    y = []
    for pair in npi_pairs_data.keys():
        npi1, npi2 = pair
        npi1_name_data = npi_name_data[npi1]['name']
        npi1_other_name_data = npi_name_data[npi1]['other']
        npi2_name_data = npi_name_data[npi2]['name']
        npi2_other_name_data = npi_name_data[npi2]['other']
        X_left.append(npi1_name_data + npi1_other_name_data)
        X_right.append(npi2_name_data + npi2_other_name_data)
        y.append(npi_pairs_data[pair])
    X = {'left': X_left, 'right': X_right}
    return X, y

def prepare_dataset(npi_name_data, train_npi_pairs, test_npi_pairs):
    (X_train, y_train) = get_npipairs_data_matrix(train_npi_pairs, npi_name_data)
    (X_test, y_test) = get_npipairs_data_matrix(test_npi_pairs, npi_name_data)
    y_train = to_categorical(np.array(y_train))
    y_test = to_categorical(np.array(y_test))
    return (X_train, y_train), (X_test, y_test)

def exponent_neg_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))

def run_malstm(X_train, y_train, max_name_len, max_other_name_len, embedding_matrix, embedding_dim, lstm_layer_size,
               loss_func='binary_crossentropy', learning_rate=0.01, X_test=None, y_test=None):
    left_input = Input(shape=(max_name_len + max_other_name_len,), dtype='int32')
    right_input = Input(shape=(max_name_len + max_other_name_len,), dtype='int32')
    embedding_layer = Embedding(len(embedding_matrix), embedding_dim, weights=[embedding_matrix],
                                input_length=max_name_len + max_other_name_len, trainable=False)
    encoded_name_left = embedding_layer(left_input[:max_name_len])
    encoded_other_name_left = embedding_layer(left_input[max_name_len:])
    encoded_name_right = embedding_layer(right_input[:max_name_len])
    encoded_other_name_right = embedding_layer(right_input[max_name_len:])

    shared_name_lstm = LSTM(lstm_layer_size)
    shared_other_name_lstm = LSTM(lstm_layer_size)

    left_name_output = shared_name_lstm(encoded_name_left)
    right_name_output = shared_name_lstm(encoded_name_right)
    left_other_name_output = shared_other_name_lstm(encoded_other_name_left)
    right_other_name_output = shared_other_name_lstm(encoded_other_name_right)

    left_output = concatenate([left_name_output, left_other_name_output])
    right_output = concatenate([right_name_output, right_other_name_output])

    malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
                             output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    model = Model([left_input, right_input], [malstm_distance])
    opt = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=opt, loss=loss_func, metrics=['mse'])

    batch_size = 128
    num_epoch = 100
    if X_test is not None:
        history = model.fit([X_train['left'], X_train['right']], y_train, batch_size=batch_size, epochs=num_epoch,
                        verbose=1, validation_data=(X_test, y_test))
        score = model.evaluate(X_test, y_test, verbose=0)
        yhat = model.predict(X_test, verbose=0)
        auc_score = roc_auc_score(y_test, yhat)
        score.append(auc_score)
        print('Current fold AUC: ', score[2])
        return score
    else:
        history = model.fit([X_train['left'], X_train['right']], y_train, batch_size=batch_size, epochs=num_epoch,
                            verbose=1)
        return model