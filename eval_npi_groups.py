from sklearn.metrics import accuracy_score
import argparse
import csv

def calculate_accuracy(pairwise_gt_file, groups_file):
    groups = []
    ytrue = []
    ypred = []
    with open(groups_file, 'r') as gr:
        groups_reader = csv.reader(gr)
        next(groups_reader)
        for row in groups_reader:
            groups.append((row[0], row[1]))
    with open(pairwise_gt_file, 'r') as pgt:
        gt_file_reader = csv.reader(pgt)
        next(gt_file_reader)
        for row in gt_file_reader:
            label = int(row[0])
            p1 = row[1]
            p2 = row[2]
            ytrue.append(label)
            if (p1, p2) in groups or (p2, p1) in groups:
                ypred.append(1)
            else:
                ypred.append(0)
    print("Accuracy score: " + str(accuracy_score(ytrue, ypred)))

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-gt', '--ground_truth', help='Path to pairwise NPI group ground truth file')
    arg_parser.add_argument('-ng', '--npi_groups', help='Path to NPI groups file')
    args = vars(arg_parser.parse_args())
    gt_file = args['ground_truth']
    groups_file = args['npi_groups']
    calculate_accuracy(gt_file, groups_file)

if __name__ == '__main__':
    main()