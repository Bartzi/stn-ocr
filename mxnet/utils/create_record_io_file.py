import argparse
import csv
import random

import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gt_file", help="path to original csv file that shall be transformed")
    parser.add_argument("dest_file", help='path to file that shall be produced')
    parser.add_argument("--shuffle", action='store_true', default=False, help='shuffle labels')
    parser.add_argument("--delimiter", default=',', help='delimiter used in original csv file')

    args = parser.parse_args()

    data = []
    with open(args.gt_file) as gt:
        reader = csv.reader(gt, delimiter='\t')
        for idx, line in enumerate(reader):
            file_name = line[0]
            labels = line[1:]

            if args.shuffle:
                random.shuffle(labels)

            data.append((idx, *labels, os.path.join(os.path.dirname(args.gt_file), file_name)))

    with open(args.dest_file, 'w') as dest_file:
        writer = csv.writer(dest_file, delimiter='\t')
        writer.writerows(data)