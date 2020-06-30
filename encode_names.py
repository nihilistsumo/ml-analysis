from nltk.tokenize import word_tokenize
import string
import math
import numpy as np
import argparse
import pandas as pd
import json
from sklearn.feature_extraction.text import HashingVectorizer

PROVIDER_NAME_COL = "Provider Organization Name (Legal Business Name)"
OTHER_NAME_COL = "Provider Other Organization Name"
VEC_LEN = 20

def build_token_dict(df):
    names = [n.translate(str.maketrans('', '', string.punctuation)) for n in df[PROVIDER_NAME_COL]]
    for n in df[OTHER_NAME_COL]:
        if not isinstance(n, float):
            names.append(n.translate(str.maketrans('', '', string.punctuation)))
    token_dict = set()
    for n in names:
        for s in word_tokenize(n):
            token_dict.add(s)
    token_dict = list(token_dict)
    return token_dict

def encode_name(n, token_dict):
    encoding = [-1] * VEC_LEN
    if not isinstance(n, float):
        n = n.translate(str.maketrans('', '', string.punctuation))
        print(n)
        tokens = word_tokenize(n)
        if len(tokens) > VEC_LEN:
            tokens = tokens[:VEC_LEN]
        for i in range(len(tokens)):
            t = tokens[i]
            if t in token_dict:
                encoding[i] = token_dict.index(t)
    return encoding

def _encode_name_hash(n, hasher):
    encoding = [-1] * VEC_LEN
    if not isinstance(n, float):
        n = n.translate(str.maketrans('', '', string.punctuation))
        print(n)
        encoding = hasher.transform([n]).toarray().tolist()[0]
    return encoding

def encode_names_hash(df):
    hasher = HashingVectorizer(n_features=VEC_LEN)
    dfname = df[PROVIDER_NAME_COL].apply(lambda x: _encode_name_hash(x, hasher))
    dfother = df[OTHER_NAME_COL].apply(lambda x: _encode_name_hash(x, hasher))
    dfnpi = list(df['NPI'])
    npi_encoded_dict = {}
    for i in range(len(dfnpi)):
        npi_encoded_dict[dfnpi[i]] = {'en_name': dfname[i], 'en_other': dfother[i]}
    return npi_encoded_dict

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-nd', '--npi_data', help='Path to NPI data')
    arg_parser.add_argument('-op', '--out', help='Path to output file')
    args = vars(arg_parser.parse_args())
    npi_file = args['npi_data']
    outfile = args['out']
    df = pd.read_csv(npi_file)
    # td = build_token_dict(df)
    hasher = HashingVectorizer(n_features=VEC_LEN)
    dfname = df[PROVIDER_NAME_COL].apply(lambda x: _encode_name_hash(x, hasher))
    dfother = df[OTHER_NAME_COL].apply(lambda x: _encode_name_hash(x, hasher))
    dfnpi = list(df['NPI'])
    npi_encoded_dict = {}
    for i in range(len(dfnpi)):
        npi_encoded_dict[dfnpi[i]] = {'en_name': dfname[i], 'en_other': dfother[i]}
    with open(outfile, 'w') as o:
        json.dump(npi_encoded_dict, o)

if __name__ == '__main__':
    main()