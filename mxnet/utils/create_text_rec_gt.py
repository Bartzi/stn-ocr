import argparse
import csv
import json


def get_gt_item(gt_file, delimiter, char_map, max_len, label_length, blank_symbol):
    if label_length is None:
        label_length = max_len
    else:
        assert label_length >= max_len, "Label length must be larger than max length"

    with open(gt_file) as the_file:
        while True:
            try:
                line = the_file.readline()
                if line == '':
                    break
                file_path, word = line.split(delimiter)
                file_path = file_path.strip()
                word = word.strip()
                characters = word.split(' ')
                characters = [character if character != 'SP' else ' ' for character in characters]
                if len(characters) > max_len:
                    continue
                labels = [char_map[ord(character.lower())] for character in characters]
                padding = [char_map[blank_symbol]] * (label_length - len(labels))
                labels.extend(padding)
                yield file_path, labels
            except (UnicodeDecodeError, ValueError, KeyError) as e:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tool that takes caffe GT for text recogntion and creates GT for MXNet')
    parser.add_argument('caffe_gt', help='path to caffe gt file')
    parser.add_argument('dest_file', help='path to resulting file')
    parser.add_argument('char_map', help='path to json file containing char_map')
    parser.add_argument('--delimiter', default='##', help='delimiter in gt file')
    parser.add_argument('--max-length', type=int, default=23, help='max length of single word')
    parser.add_argument('--label-length', type=int, help='length of labels for each word')
    parser.add_argument('--blank_symbol', type=int, default=9250, help='char code of blank symbol')

    args = parser.parse_args()

    with open(args.char_map) as the_file:
        char_map = json.load(the_file)

    reverse_char_map = {v: k for k, v in char_map.items()}

    with open(args.dest_file, 'w') as result_file:
        writer = csv.writer(result_file, delimiter='\t')
        iterator = get_gt_item(args.caffe_gt, args.delimiter, reverse_char_map, args.max_length, args.label_length, args.blank_symbol)
        for file_path, labels in iterator:
            writer.writerow([file_path] + labels)

