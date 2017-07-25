import argparse
import csv

import itertools


def split_list(the_list, parts):
    length = len(the_list)
    splits = [the_list[i * length // parts: (i + 1) * length // parts] for i in range(parts)]
    return splits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that takes a lst file for mxnet record io generation and reorders the label positions')
    parser.add_argument('lst_file', help='path to the file where data shall be reordered')
    parser.add_argument('--num-labels', type=int, help='label count in file', required=True)
    parser.add_argument('--num-groups', type=int, help='number of label groups (i.e. how many labels describe on entity', required=True)
    parser.add_argument('--group-order', help='new order of label groups (i.e. 3,2,1,0 to reverse a file containing four groups', required=True)

    args = parser.parse_args()

    group_order = [int(x) for x in args.group_order.split(',')]

    data = []
    with open(args.lst_file) as list_file:
        reader = csv.reader(list_file, delimiter='\t')
        for line in reader:
            index = line[0]
            labels = line[1:args.num_labels + 1]
            file_name = line[-1]

            groups = split_list(labels, args.num_groups)
            new_group_list = [groups[index] for index in group_order]
            new_labels = list(itertools.chain(*new_group_list))

            data.append([
                index,
                *new_labels,
                file_name
            ])

    with open(args.lst_file, 'w') as list_file:
        writer = csv.writer(list_file, delimiter='\t')
        writer.writerows(data)
