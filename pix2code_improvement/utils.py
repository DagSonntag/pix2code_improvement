import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
from constants import START_WORD, END_WORD, PLACEHOLDER


def preprocess_image(img_path: str or Path, img_shape: int or Tuple[int]) -> np.array:
    """
    Loads an image and does the preprocessing used by the models
    :param img_path: The path to the image file
    :param img_shape: The desired shape or size of the image
    :return: A numpy array of shape (img_shape[0], img_shape[1],3)
    """
    if isinstance(img_shape, int):
        img_shape = (img_shape, img_shape)
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, img_shape)
    img = img.astype('float32')
    img /= 255
    return img


def load_code(file_path: str or Path, clean_code_on_load=True) -> str:
    """
    Loads the code from a given gui file path
    :param file_path: The path to the file
    :param clean_code_on_load: Whether or not to clean the code
    :return: The (cleaned) code represented in the file
    """
    with open(str(file_path), 'r') as f:
        gui_code = f.read()
    return clean_code(gui_code) if clean_code_on_load else gui_code


def clean_code(code: str) -> str:
    """
    Cleans the code by removing unnecessary chars
    :param code: The unclean string
    :return: The cleaned string
    """
    code = code.replace(",", " ").replace("\n", " ").replace("  ", " ")
    code = code.replace("{", 'square_bracket').replace("}", 'close_square_bracket')
    if len(code) > 0 and code[-1] == " ":
        code = code[:-1]
    return code


def preprocess_code2context(code, include_context_mode, voc, fixed_output_length, word_index):
    code_len = len(code.split(" ")) + 1
    if include_context_mode == 'single_word':
        if fixed_output_length is None:
            raise ValueError("If include_context_mode is set to single word the fixed_output_length "
                             "must be set")
        context_words = [PLACEHOLDER] * fixed_output_length + [START_WORD] + code.split(" ")
        context_words = context_words[(word_index + 1):(word_index + fixed_output_length + 1)]
        context_one_hot = np.array(voc.one_hot_encode(context_words))
        return context_one_hot
    elif include_context_mode == 'full_code':
        code_one_hot = np.array(voc.one_hot_encode(START_WORD + " " + code + " " + END_WORD))
        if fixed_output_length is not None:
            if code_len > fixed_output_length:
                raise ValueError("Code found with larger size than output")
            code_array = np.array(voc.one_hot_encode(PLACEHOLDER) * fixed_output_length)
            code_array[:code_one_hot.shape[0], :] = code_one_hot
            return code_array
        else:
            return code_one_hot
    else:
        raise ValueError("Unknown include_context_mode. Possible values are "
                         "['single_word', 'full_code']")

