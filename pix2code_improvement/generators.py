__author__ = 'Dag Sonntag'

import numpy as np
import tensorflow as tf
from typing import List
import logging

from constants import START_WORD, END_WORD, PLACEHOLDER, IMAGE_SIZE
from vocabulary import Vocabulary
from utils import preprocess_code2context

from tensorflow.keras.preprocessing.image import ImageDataGenerator

logger = logging.getLogger()


def features_only(x, y):
    return x


def single_instance_generator_creator(npz_file_paths: List[str],
                                      voc: Vocabulary,
                                      include_context_mode: str or None = None,
                                      include_code_length: bool = False,
                                      include_image_object_count_words: List[str] or None = None,
                                      include_code_mode: str or None = None,
                                      fixed_output_length: int or None = None,
                                      image_transformer: ImageDataGenerator = None):
    """
    A generator creator that yields single instance x and y dictionaries for the dataset creation. The x always includes
    the img_data feature, and the rest are set according to the input. For the code and context features different
    modes can be set to allow different functionality. The modes are ['single_word', 'full_code'] and work as
    follows:
    single_word: Supplies data for the single word prediction use case. I.e. supplies a one hot encoding of a
    randomly selected word as "code" and the previous fixed_output_length number of words as context (shape
    [fixed_output_length, voc.size]) Padding is used to fill "empty" positions and <sos> and <eos> words are added
    as start of sequence and end of sequence words.
    full_code: Supplies data for the full code prediction use case. I.e. supplies all one hot encoded words as y
    (shape [fixed_output_length, voc.size] if fixed_output_length is not null, else [len(code_words+1), voc.size]).
    As context the same data is supplied shifted one position with <sos> added as first token while code has <eos>
    added as last token.

    Note that this is just the creator, and that the actual generator has has to be called on the return value. This
    setup is necessary to circumvent the limitations when using a dataset from generator pipeline with custom
    classes (since everything must be converted to tensors).
    :param npz_file_paths: The paths to the npz files that are to be used by the generator
    :param voc: The vocabulary used for the dataset
    :param include_context_mode: If the context should be included in the x variable (as context) and its mode.
    The possible modes are: ['single_word', 'full_code'] as detailed above.
    :param include_code_length: If the expected code length (including eos) should be supplied in x (as code_length)
    :param include_image_object_count_words: The image objects that should be included in y as "word"_count.
    :param include_code_mode: If the code should be included in the output (as code) and its mode.
    The possible modes are: ['single_word', 'full_code'] as detailed above.
    :param fixed_output_length: The length of the code and context settings as detailed above.
    :param image_transformer: If supplied, an image transformer that is applied on the images before usage
    :return: A generator yielding dictionaries x and y with the appropriate keys
    """
    def single_instance_generator():
        while True:
            for npz_path in npz_file_paths:
                data_file = np.load(npz_path, allow_pickle=True)
                img_data = data_file['img_data']
                code = str(data_file['code'])

                if image_transformer is not None:
                    img_data = image_transformer.random_transform(img_data)

                x = {'img_data': tf.convert_to_tensor(img_data)}
                y = {}
                code_len = len(code.split(" "))+1
                word_index = np.random.randint(code_len)
                # x: Code length handling
                if include_code_length:
                    x.update({'code_length': tf.convert_to_tensor(code_len)})
                # x: Context handling:
                if include_context_mode is not None:
                    context = preprocess_code2context(code, include_context_mode, voc, fixed_output_length, word_index)
                    x.update({'context': tf.convert_to_tensor(context)})

                # y: code handling
                if include_code_mode is not None:
                    if include_context_mode == 'single_word':
                        if fixed_output_length is None:
                            raise ValueError("If include_code_mode is set to single word the fixed_output_length "
                                             "must be set")
                        code_words = code.split(" ") + [END_WORD]
                        code_word = code_words[word_index]
                        y.update({'code': tf.convert_to_tensor(voc.one_hot_encode(code_word)[0])})
                    elif include_code_mode == 'full_code':
                        code_one_hot = np.array(voc.one_hot_encode(code + " " + END_WORD))
                        if fixed_output_length is not None:
                            if code_len > fixed_output_length:
                                raise ValueError("Code found with larger size than output: {}".format(npz_path))
                            code_array = np.array(voc.one_hot_encode(PLACEHOLDER) * fixed_output_length)
                            code_array[:code_one_hot.shape[0], :] = code_one_hot
                            y.update({'code': tf.convert_to_tensor(code_array)})
                        else:
                            y.update({'code': tf.convert_to_tensor(code_one_hot)})
                    else:
                        raise ValueError("Unknown include_code_mode. Possible values are ['single_word', 'full_code']")
                # y: Image counts handling
                if include_image_object_count_words is not None:
                    y.update({key: tf.convert_to_tensor(val, dtype=tf.float32) for key, val
                             in voc.count_words(code, words_to_count=include_image_object_count_words).items()})
                yield x, y
    return single_instance_generator


def dataset_generator_reader(npz_file_paths: List[str],
                             voc: Vocabulary,
                             include_context_mode: str or None = None,
                             include_code_length: bool = False,
                             include_image_object_count_words: List[str] or None = None,
                             include_code_mode: str or None = None,
                             fixed_output_length: int or None = None,
                             image_transformer=None,
                             batch_size: int = 32,
                             prefetch: int = 1,
                             image_size: int = IMAGE_SIZE):
    """
   A dataset creator that creator that returns a dataset of x and y dictionaries for the pix2code setup.
   The x always includes the img_data feature, and the rest are set according to the input.
   For the code and context features different modes can be set to allow different functionality.
   The modes are ['single_word', 'full_code'] and work as follows:
   - single_word: Supplies data for the single word prediction use case. I.e. supplies a one hot encoding of a
   randomly selected word as "code" and the previous fixed_output_length number of words as context (shape
   [fixed_output_length, voc.size]) Padding is used to fill "empty" positions and <sos> and <eos> words are added
   as start of sequence and end of sequence words.
   - full_code: Supplies data for the full code prediction use case. I.e. supplies all one hot encoded words as y
   (shape [fixed_output_length, voc.size] if fixed_output_length is not null, else [len(code_words+1), voc.size]).
   As context the same data is supplied shifted one position with <sos> added as first token while code has <eos>
   added as last token.

   :param npz_file_paths: The paths to the npz files that are to be used by the generator
   :param voc: The vocabulary used for the dataset
   :param include_context_mode: If the context should be included in the x variable (as context) and its mode.
   The possible modes are: ['single_word', 'full_code'] as detailed above.
   :param include_code_length: If the expected code length (including eos) should be supplied in x (as code_length)
   :param include_image_object_count_words: The image objects that should be included in y as "word"_count.
   :param include_code_mode: If the code should be included in the output (as code) and its mode.
   The possible modes are: ['single_word', 'full_code'] as detailed above.
   :param fixed_output_length: The length of the code and context settings as detailed above.
   :param image_transformer: If supplied, an image transformer that is applied on the images before usage
   :param batch_size: The batch size to use
   :param prefetch: The amount of batches to load in advance
   :param image_size: The size of the images saved in the npz files

   :return: A Tensorflow dataset of x and y

   """

    generator_creator = single_instance_generator_creator(
        npz_file_paths=npz_file_paths, voc=voc, include_context_mode=include_context_mode,
        include_code_length=include_code_length, include_image_object_count_words=include_image_object_count_words,
        include_code_mode=include_code_mode, fixed_output_length=fixed_output_length,
        image_transformer=image_transformer)

    output_types_x = {'img_data': tf.float32}
    output_shapes_x = {'img_data': tf.TensorShape([image_size, image_size, 3])}
    if include_code_length:
        output_types_x.update({'code_length': tf.int32})
        output_shapes_x.update({'code_length': tf.TensorShape([])})

    if include_context_mode is not None:
        output_types_x.update({'context': tf.float32})
        if include_context_mode == 'single_word':
            if fixed_output_length is None:
                raise ValueError("If include_context_mode is set to single word the fixed_output_length "
                                 "must be set")
            output_shapes_x.update({'context': tf.TensorShape([fixed_output_length, voc.size])})
        elif include_context_mode == 'full_code':
            if fixed_output_length is not None:
                output_shapes_x.update({'context': tf.TensorShape([fixed_output_length, voc.size])})
            else:
                output_shapes_x.update({'context': tf.TensorShape([None, voc.size])})

    output_types_y = {}
    output_shapes_y = {}
    if include_image_object_count_words is not None:
        output_types_y.update({key + "_count": tf.float32 for key in include_image_object_count_words})
        output_shapes_y.update({key + "_count": tf.TensorShape([]) for key in include_image_object_count_words})

    if include_code_mode is not None:
        output_types_y.update({'code': tf.float32})
        if include_code_mode == 'single_word':
            output_shapes_y.update({'code': tf.TensorShape([voc.size])})
        elif include_code_mode == 'full_code':
            if fixed_output_length is not None:
                output_shapes_y.update({'code': tf.TensorShape([fixed_output_length, voc.size])})
            else:
                output_shapes_y.update({'code': tf.TensorShape([None, voc.size])})
    dataset = tf.data.Dataset.from_generator(generator_creator,
                                             output_types=(output_types_x, output_types_y),
                                             output_shapes=(output_shapes_x, output_shapes_y))
    return dataset.batch(batch_size).prefetch(prefetch)
