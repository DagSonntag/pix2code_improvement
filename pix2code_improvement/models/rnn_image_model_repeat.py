import tensorflow as tf

from tensorflow.python.keras.layers import Dense, Dropout, \
    RepeatVector, concatenate, Flatten

from tensorflow.keras.optimizers import RMSprop
from utils import preprocess_image
import numpy as np
from constants import START_WORD, END_WORD, PLACEHOLDER, IMAGE_SIZE
from models.shallow_cnn_model import ShallowCnnUnit, CounterUnit


class RnnImageModelRepeat(tf.keras.models.Model):

    def __init__(self,  words, image_count_words, *args, max_code_length, activation='relu',
                 order_layer_output_size=1024, kernel_shape=7, dropout_ratio=0.25, dense_layer_size=512,
                 image_out=False, **kwargs):

        super().__init__(*args, **kwargs)
        self.image_out = image_out

        self.voc_size = len(words)
        self.image_out = image_out
        self.layer_output_names = words
        self.image_count_words = image_count_words
        self.max_code_length = max_code_length

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

        self.repeat_image_layer = RepeatVector(max_code_length)

        self.language_model_layers = [
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(128, return_sequences=True)
        ]

        self.decoder_layers = [
            tf.keras.layers.LSTM(512, return_sequences=True),
            tf.keras.layers.LSTM(512, return_sequences=True),
            Dense(len(words), activation='softmax')
        ]

    def call(self, inputs):
        inp = inputs['img_data']
        context_inp = inputs['context']

        cnn_output_layers = self.shallow_cnn_unit(inp)
        obj_counter_ouputs = [self.parallel_counter_unit(img_out) for img_out in cnn_output_layers]
        obj_counter_ouputs = {key + "_count": layer for key, layer in zip(self.image_count_words, obj_counter_ouputs)}

        ordering_inp = concatenate(cnn_output_layers)
        for layer in self.ordering_layers:
            ordering_inp = layer(ordering_inp)

        img_out = self.repeat_image_layer(ordering_inp)

        for layer in self.language_model_layers:
            context_inp = layer(context_inp)

        decoder = concatenate([img_out, context_inp])
        for layer in self.decoder_layers:
            decoder = layer(decoder)

        outputs = obj_counter_ouputs
        outputs.update({'code': decoder})

        if self.image_out:
            outputs.update({"img_out_" + key: layer for key, layer in zip(self.image_count_words,
                                                                          cnn_output_layers)})
        return outputs

    def compile(self, loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, clipvalue=1.0), **kwargs):
        # Fix the bug in tensorflow 1.15 that sets the outputnames wrong when using dicts and generators
        if self.image_out:
            names = ([key + "_count" for key in self.image_count_words]
                     + ["img_out_" + key for key in self.image_count_words]
                     + ["code"])
        else:
            names = [key + "_count" for key in self.image_count_words] + ['code']
        self.output_names = sorted(names)

        return super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def predict(self, *args, return_as_dict=True, **kwargs):
        pred = super().predict(*args, **kwargs)
        if return_as_dict:
            return {key: val for key, val in zip(self.output_names, pred)}
        else:
            return pred

    def predict_image(self, image, voc, img_size=IMAGE_SIZE):

        if isinstance(image, str):
            img = preprocess_image(image, img_size)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("Unknown handling of image input of type {}".format(type(image)))

        img_inp = np.expand_dims(img, 0)

        current_context = np.array([voc.word2one_hot_dict[PLACEHOLDER]] * (self.max_code_length))
        current_context = np.expand_dims(current_context, 0)
        current_context[0, 0, :] = voc.word2one_hot_dict[START_WORD]

        word_predictions = []
        for i in range(self.max_code_length):
            probas = self.predict({'img_data': img_inp, 'context': current_context})
            probas = probas['code'][0][i]
            prediction = np.argmax(probas)
            word_prediction = voc.token2word_dict[prediction]
            if word_prediction == END_WORD or word_prediction == PLACEHOLDER:
                break
            word_predictions.append(word_prediction)
            current_context[0, i + 1, :] = voc.word2one_hot_dict[word_prediction]

        return " ".join(word_predictions)
