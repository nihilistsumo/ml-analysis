import re
import pandas as pd

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
    corpus_lines = []
    for name_field in name_corpus:
        tokens = []
        for token in name_field:
            tokens.append(token)
        corpus_lines.append(tokens)
    with (output_file, 'w') as out:
        out.write(corpus_lines)