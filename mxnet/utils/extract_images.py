import argparse
import csv

import os

import numpy as np
from PIL import Image

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument("npz_file", help="path to npz where images shall be extracted")
    parser.add_argument("dest_dir", help='path to directory where extracted images and gt shall be placed')
    parser.add_argument("-s", "--shape", type=int, default=100, help="shape of images in shape x shape")

    args = parser.parse_args()

    data = np.load(args.npz_file)
    files = data.files
    image_files = sorted(filter(lambda x: "x" in x.lower(), files))
    label_files = sorted(filter(lambda x: "y" in x.lower(), files))

    for image_file, label_file in zip(image_files, label_files):
        images = data[image_file]
        labels = data[label_file]
        image_stage = image_file.split("_")[-1]
        label_stage = label_file.split("_")[-1]
        assert image_stage == label_stage

        os.makedirs(os.path.join(args.dest_dir, image_stage), exist_ok=True)
        out_data = []
        print("working on stage: {}".format(image_stage))

        for idx, (image_data, label_data) in enumerate(zip(images, labels)):
            image_data *= 255
            image = Image.fromarray(image_data.reshape((args.shape, args.shape)).astype(np.uint8))
            image = image.convert('RGB')
            image_path = os.path.join(image_stage, "{}.png".format(idx))
            image.save(os.path.join(args.dest_dir, image_path))
            out_data.append([image_path, *label_data])

        with open(os.path.join(args.dest_dir, "{}_gt.txt".format(label_stage)), "w") as the_label_file:
            writer = csv.writer(the_label_file, delimiter=';')
            writer.writerows(out_data)
