import cv2
import numpy as np
import concurrent.futures


def load_image(path):
    return cv2.imread(path)


def load_image_with_channel(path):
    return cv2.imread(path, 0)


def save_images(chars, destination, word_index):
    for index, char in enumerate(chars):
        char = cv2.resize(char, (28, 28))
        cv2.imwrite(str(destination) + '/' + str(str(word_index) + "_" + str(index) + ".png"), char)


def get_words(image):
    words = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5, 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])
    for line_index, ctr in enumerate(sorted_ctrs):

        x, y, w, h = cv2.boundingRect(ctr)

        row = image[y:y + h, x:x + w]

        ret_row, thresh_row = cv2.threshold(row, 127, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((15, 15), np.uint8)
        dilated_lines = cv2.dilate(thresh_row, kernel, iterations=1)

        dilated_words = cv2.cvtColor(dilated_lines, cv2.COLOR_BGR2GRAY)

        ctrs_words, hier_words = cv2.findContours(dilated_words, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        sorted_ctrs_words = sorted(ctrs_words, key=lambda ctr: cv2.boundingRect(ctr)[0])

        for word_index, ctr_word in enumerate(sorted_ctrs_words):

            x_word, y_word, w_word, h_word = cv2.boundingRect(ctr_word)
            word = row[y_word:y_word + h_word, x_word:x_word + w_word]

            ret_word, thresh_word = cv2.threshold(word, 127, 255, cv2.THRESH_BINARY_INV)

            thresh_word = cv2.cvtColor(thresh_word, cv2.COLOR_BGR2GRAY)

            ctrs_chars, hier_chars = cv2.findContours(thresh_word, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sorted_ctrs_chars = sorted(ctrs_chars, key=lambda ctr: cv2.boundingRect(ctr)[0])

            chars = []

            for char_index, ctr_char in enumerate(sorted_ctrs_chars):
                x_char, y_char, w_char, h_char = cv2.boundingRect(ctr_char)
                # print(x_char, y_char, w_char, h_char)

                if w_char <= 7 or h_char <= 5:
                    continue

                char = word[y_char:y_char + h_char, x_char:x_char + w_char]
                cv2.resize(char, (0, 0), fx=2, fy=2)
                char = cv2.copyMakeBorder(char, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                char = 255 - char
                dilated_char = cv2.dilate(char, (2, 2))
                char_erode = cv2.erode(dilated_char, (1, 1))

                chars.append(255 - char_erode)

            if len(chars) != 0:
                words.append(chars)

    return words

def process(image, destination):
    words = get_words(image)
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for index, chars in enumerate(words):
            executor.submit(save_images, chars, destination, index + 1)