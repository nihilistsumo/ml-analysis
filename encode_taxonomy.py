from sklearn.preprocessing import LabelEncoder
import json

TAXONOMY_COL = 'Healthcare Provider Taxonomy Code_'

def get_taxonomy_codes(df):
    cols = [TAXONOMY_COL + str(i) for i in range(1, 16)]
    taxonomy_dict = df[['NPI']+cols].set_index('NPI').to_dict()
    return taxonomy_dict

def encode_taxonomy_codes(taxonomy_file, output_file):
    tax_code = []
    tx1 = []
    tx2 = []
    tx3 = []
    with open(taxonomy_file, 'r') as tf:
        for l in tf:
            tax_code.append(l.split(',')[0])
            tx1.append(l.split(',')[1])
            tx2.append(l.split(',')[2])
            tx3.append(l.split(',')[3])
    le = LabelEncoder()
    tx1_label = le.fit_transform(tx1)
    tx2_label = le.fit_transform(tx2)
    tx3_label = le.fit_transform(tx3)
    taxonomy_labeled = {}
    for i in range(len(tax_code)):
        taxonomy_labeled[tax_code[i]] = (int(tx1_label[i]), int(tx2_label[i]), int(tx3_label[i]))
    with open(output_file, 'w') as out:
        json.dump(taxonomy_labeled, out)