import geolocate_address
import encode_names
import encode_taxonomy
import pandas as pd
import json

def embed_all_npis(npidata_file, exact_loc_file, zip_loc_file, encoded_taxonomy_file, out_embed_file):
    df = pd.read_csv(npidata_file)
    npis = list(df['NPI'])
    loc_dict = geolocate_address.get_npi_coordinates_from_file(npis, exact_loc_file, zip_loc_file)
    tax_dict = encode_taxonomy.get_taxonomy_codes(df)
    names_dict = encode_names.encode_names_hash(df)
    with open(encoded_taxonomy_file, 'r') as ef:
        tax_code_label_dict = json.load(ef)
    npi_embed = {}
    for n in npis:
        embed_dat = {}
        embed_dat['name'] = names_dict[n]['en_name']
        embed_dat['other'] = names_dict[n]['en_other']
        embed_dat['location'] = loc_dict[n]
        taxonomy = []
        for i in range(1, 16):
            code = tax_dict[encode_taxonomy.TAXONOMY_COL+str(i)][n]
            if isinstance(code, float):
                taxonomy.append([-1, -1, -1])
            else:
                taxonomy.append(tax_code_label_dict[code])
        embed_dat['taxonomy'] = taxonomy
        npi_embed[n] = embed_dat
    with open(out_embed_file, 'w') as out:
        json.dump(npi_embed, out)