import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, concatenate
from tensorflow.keras.optimizers import RMSprop
from constants import START_WORD, END_WORD, PLACEHOLDER, IMAGE_SIZE, CONTEXT_LENGTH
from utils import preprocess_image
import numpy as np
from .shallow_cnn_model import ShallowCnnUnit, CounterUnit


class ShallowImageModel(tf.keras.Model):
    """
    A model with the following structure:
    input: image-array
    output: 1. The counts of each object, 2. The 50 first words of the code
    Layers
    - CNN-layers: standard downsampling from 256 to 64 pix
    - 1 set of parallel layers of size (voc-size) 1 node each - named appropriately
    - Parallel counting units to each of these
    - 1 ordering unit to all the concatenated layers consisting of 1024+1024 and then 50 parallel softmax layers

    """

    def __init__(self, words, image_count_words, max_sentence_length=100, kernel_shape=7, dense_layer_size=256, activation='relu',
                 dropout_ratio=0.25, order_layer_output_size=1024,
                 image_out=False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        if isinstance(kernel_shape, tuple) or isinstance(kernel_shape, list):
            kernel_shape = tuple(kernel_shape)
        else:
            kernel_shape = (kernel_shape, kernel_shape)

        self.voc_size = len(words)
        self.image_out = image_out
        self.layer_output_names = words
        self.image_count_words = image_count_words
        self.max_sentence_length = max_sentence_length

        self.shallow_cnn_unit = ShallowCnnUnit(image_count_words=image_count_words, kernel_shape=kernel_shape,
                                               dropout_ratio=dropout_ratio, activation=activation, name='cnn_unit')
        self.parallel_counter_unit = CounterUnit(layer_size=dense_layer_size, activation=activation,
                                                 name='counter_unit')

        self.ordering_layers = [
            Flatten(name='ordering_flatten'),
            Dense(1024, activation=activation, name='ordering_1'),
            Dropout(dropout_ratio, name='ordering_drop_1'),
            Dense(1024, activation=activation, name='ordering_2'),
            Dropout(dropout_ratio, name='ordering_drop_2'),
            Dense(order_layer_output_size, activation=activation, name='ordering_3')
        ]
        self.code_output_units = [Dense(self.voc_size, activation='softmax', name="code_out_{}".format(i))
                                  for i in range(max_sentence_length)]

    def call(self, inputs):
        inp = inputs['img_data']
        cnn_output_layers = self.shallow_cnn_unit(inp)
        obj_counter_ouputs = [self.parallel_counter_unit(img_out) for img_out in cnn_output_layers]
        obj_counter_ouputs = {key + "_count": layer for key, layer in zip(self.image_count_words, obj_counter_ouputs)}

        ordering_inp = concatenate(cnn_output_layers)
        for layer in self.ordering_layers:
            ordering_inp = layer(ordering_inp)
        code_ordered_out_layers = [layer(ordering_inp) for layer in self.code_output_units]
        code = concatenate([tf.expand_dims(l, 1) for l in code_ordered_out_layers], axis=1)

        outputs = obj_counter_ouputs
        outputs.update({'code': code})

        if self.image_out:
            outputs.update({"img_out_" + key: layer for key, layer in zip(self.image_count_words,
                                                                          cnn_output_layers)})
        return outputs

    def compile(self, loss='mse', optimizer=RMSprop(lr=0.0001, clipvalue=1.0), *args, **kwargs):
        # Fix the bug in tensorflow 1.15 that sets the outputnames wrong when using dicts and generators
        if self.image_out:
            names = ([key + "_count" for key in self.image_count_words]
                     + ["img_out_" + key for key in self.image_count_words]
                     + ["code"])
        else:
            names = [key + "_count" for key in self.image_count_words] + ['code']
        self.output_names = sorted(names)
        return super().compile(loss=loss, optimizer=optimizer, *args, **kwargs)

    def predict(self, *args, return_as_dict=True, **kwargs):
        pred = super().predict(*args, **kwargs)
        if return_as_dict:
            return {key: val for key, val in zip(self.output_names, pred)}
        else:
            return pred

    def predict_image(self, image, voc, img_size=IMAGE_SIZE, context_length=CONTEXT_LENGTH):

        def clean_prediction(pred, voc):
            pred_code_one_hot = pred['code'][0]
            pred_code_tokens = np.argmax(pred_code_one_hot, axis=1)

            end_index = np.where(pred_code_tokens == voc.word2token_dict[END_WORD])
            if len(end_index[0]) == 0:
                end_index = np.where(pred_code_tokens == voc.word2token_dict[PLACEHOLDER])
                if len(end_index[0]) == 0:
                    end_index = pred_code_tokens.shape[0]
            end_index = end_index[0][0]
            pred_code = [voc.token2word_dict[val] for val in (pred_code_tokens[:end_index])]
            return " ".join(pred_code)

        if isinstance(image, str):
            img = preprocess_image(image, img_size)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("Unknown handling of image input of type {}".format(type(image)))

        pred = self.predict({'img_data': np.expand_dims(img, 0)})
        return clean_prediction(pred, voc)