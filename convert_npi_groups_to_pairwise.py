import csv
import model_npi_sim

def write_pairwise_npi_group_data(npi_embeddings_file, npi_groups_file, outfile):
    (npi_embed_data, npi_group_map) = model_npi_sim.get_data(npi_embeddings_file, npi_groups_file)
    train_npi_pairs = model_npi_sim.sample_npi_groups(list(npi_group_map.keys()), npi_group_map)
    with open(outfile, 'w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['label', 'npi1', 'npi2'])
        for k in train_npi_pairs.keys():
            writer.writerow([str(train_npi_pairs[k]), k[0], k[1]])