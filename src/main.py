import os
import shutil

from src.operations import *
from src.peredict import peredict

source_directory_path = "../images/"
output_directory_path = "../resources/"
file_name = "cats_story.png"


def run():
    if os.path.exists(output_directory_path):
        shutil.rmtree(output_directory_path)

    image = load_image_from_path(source_directory_path + file_name)

    os.makedirs(output_directory_path)
    progress(image, output_directory_path)

    peredict()


if __name__ == "__main__":
    run()
