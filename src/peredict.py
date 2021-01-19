import glob
from pathlib import Path

import cv2
import numpy as np
from tensorflow import keras


emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]
model = keras.models.load_model('/content/text-recognition/models/emnist_letters.h5')


def emnist_predict_img(img):
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img_arr = np.expand_dims(img, axis=0)
    img_arr = 1 - img_arr / 255.0
    img_arr[0] = np.rot90(img_arr[0], 3)
    img_arr[0] = np.fliplr(img_arr[0])
    img_arr = img_arr.reshape((1, 28, 28, 1))

    result = model.predict_classes([img_arr])
    return chr(emnist_labels[result[0]])


def get_word_char_indexes(filename):
    parts = filename.split('_')
    return int(parts[0]), int(parts[1].split('.')[0])

def peredict():
    i = 1
    line = ""
    while True:
        file_list = glob.glob("/content/text-recognition/resources/" + str(i) + "_*.png")
        if len(file_list) == 0:
            break
        j = 0
        while True:
            my_file = Path("/content/text-recognition/resources/" + str(i) + "_" + str(j) + ".png")
            if my_file.is_file():
                img = load_image_with_channel(str(my_file))
                s_out = emnist_predict_img(img)
                line = line + s_out
                j = j + 1
            else:
                break

        if i % 9 == 0:
            line = line + "\n"
        else:
            line = line + " "
        i = i + 1

    return line