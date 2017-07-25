import argparse
import os
import re

from collections import namedtuple

import subprocess

import tempfile
from PIL import Image
from PIL import ImageChops
from PIL.GifImagePlugin import getheader, getdata


SUPPORTED_IMAGETYPES = [".png", ".jpg", ".jpeg"]
ImageData = namedtuple('ImageData', ['file_name', 'path'])


def make_video(image_dir, dest_file, pattern=r"(\d+)"):
    sort_pattern = re.compile(pattern)

    image_files = filter(lambda x: os.path.splitext(x)[-1] in SUPPORTED_IMAGETYPES, os.listdir(image_dir))
    images = []

    print("loading images")
    for idx, file_name in enumerate(image_files):
        path = os.path.join(image_dir, file_name)
        images.append(ImageData(file_name=file_name, path=path))

    print("sorting images")
    images_sorted = sorted(images, key=lambda x: int(re.search(sort_pattern, x.file_name).group(1)))

    print("creating temp file")
    with tempfile.NamedTemporaryFile(mode='w') as temp_file:
        for image in images_sorted:
            print(image.path, file=temp_file)

        print("creating video")
        process_args = ['convert', '-quality 100', '@{}'.format(temp_file.name), dest_file]
        print(" ".join(process_args))
        subprocess.run(' '.join(process_args), shell=True, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool that creates a gif out of a number of given input images')
    parser.add_argument("image_dir", help="path to directory that contains all images that shall be converted to a gif")
    parser.add_argument("dest_file", help="path to destination gif file")
    parser.add_argument("--pattern", default=r"(\d+)", help="naming pattern to extract the ordering of the images")

    args = parser.parse_args()

    make_video(args.image_dir, args.dest_file, args.pattern)
