import argparse
import os

from PIL import Image
from PIL import ImageSequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool that extracts the last image from a given gif, saves the extracted image to the same location as original gif")
    parser.add_argument("gif", help='path to the gif where data shall be extracted from')

    args = parser.parse_args()

    with Image.open(args.gif) as the_gif:
        last_frame = None
        for frame in ImageSequence.Iterator(the_gif):
            last_frame = frame

        if last_frame is None:
            print("Could not find last frame")
            exit()

        last_frame.save(
            os.path.join(
                os.path.dirname(args.gif),
                "{}_last_frame.png".format(
                    os.path.splitext(os.path.basename(args.gif))[0]
                )
            )
        )
