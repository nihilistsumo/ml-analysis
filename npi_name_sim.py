import re
import pandas as pd
import numpy as np

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
    max_name_len = max([len(npi_name_data[npi]['name']) for npi in npi_name_data.keys()] +
                       [len(npi_name_data[npi]['other']) for npi in npi_name_data.keys()])
    return npi_name_data, max_name_len