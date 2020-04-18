__author__ = 'Dag Sonntag'

import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from typing import Collection, List
from constants import START_WORD, END_WORD, PLACEHOLDER
from utils import load_code


def load_voc_from_paths(paths: Collection[str]):
    """
    Creates a vocabulary from the data in a set of gui files
    :param paths: The paths to the gui files
    :return: A vocabulary
    """
    all_codes = [load_code(path) for path in paths]
    words = set(" ".join(all_codes).split(" "))
    return Vocabulary(words)


def load_voc_from_file(file_path: str):
    """
    Loads a vocabulary from file
    :param file_path: The path to the vocabulary
    :return: The vocabulary instance
    """
    with open(file_path, 'rb') as f:
        voc = pickle.load(f)
    return voc


class Vocabulary:
    """
    The core class for handling the encoding of words into tokens and one hot encodings (and vice versa)
    """
    def __init__(self, words: Collection[str]):
        """
        Creates the Vocabulary for the given words and unidentifiable words
        :param words: The words to create the Vocabulary for
        """
        self.object_words = sorted(list(words))
        self.words = [PLACEHOLDER, START_WORD, END_WORD] + self.object_words
        self.word2token_dict = {word: tok for tok, word in enumerate(self.words)}
        self.token2word_dict = {tok: word for tok, word in enumerate(self.words)}
        self.size = len(self.words)
        self.word2one_hot_dict = {word: tf.one_hot(self.word2token_dict[word], self.size).numpy()
                                  for word in self.words}

    def save(self, file_path: str) -> None:
        """
        Saves the vocabulary to the given filepath
        :param file_path: The filepath to save the vocabulary to
        :return:
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def count_words(self, sentence: List[str] or str,
                    words_to_count: List[str] or None = None) -> dict:
        """
        Counts the number of object words that occur in a list of words
        :param sentence: The sentence (words separated by spaces) or list of words to count occurrences in
        :param words_to_count: If only a subset of words in the vocabulary should be counted
        :return: An array of the counts of the object words in order
        """
        if isinstance(sentence, list):
            counts = pd.Series(sentence).value_counts()
        else:
            counts = pd.Series(sentence.split(" ")).value_counts()
        if words_to_count is None:
            words_to_count = self.words
        return {word+"_count": counts.loc[word] if word in counts.index else 0 for word in words_to_count}

    def tokenize(self, sentence: List[str] or str) -> List[int]:
        """
        Tokenizes a sentence or a list of words
        :param sentence: The sentence (words separated by spaces) or list of words to tokenize
        :return: The list of tokens representing the words in the vocabulary
        """
        if isinstance(sentence, list):
            return [self.word2token_dict[word] for word in sentence]
        else:
            return [self.word2token_dict[word] for word in sentence.split(" ")]

    def one_hot_encode(self, sentence) -> List[np.ndarray]:
        """
        Tokenizes a sentence or a list of words
        :param sentence: The sentence (words separated by spaces) or list of words to get the one hot encodings for
        :return: A list of one hot encodings of the words
        """
        if isinstance(sentence, list):
            return [self.word2one_hot_dict[word] for word in sentence]
        else:
            return [self.word2one_hot_dict[word] for word in sentence.split(" ")]
