import os
import shutil

from src.operations import *
from src.peredict import peredict

output_directory_path = "/content/text-recognition/resources"
if os.path.exists(output_directory_path):
    shutil.rmtree(output_directory_path)

os.makedirs(output_directory_path)

image = load_image("/content/text-recognition/images/cats_story.png")

process(image, output_directory_path)

peredict()