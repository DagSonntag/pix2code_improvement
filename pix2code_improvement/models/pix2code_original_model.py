import tensorflow as tf

from tensorflow.python.keras.layers import Input, Dense, Dropout, \
    RepeatVector, concatenate, \
    Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.optimizers import RMSprop
from constants import CONTEXT_LENGTH
from utils import preprocess_image
import numpy as np
from constants import START_WORD, END_WORD, PLACEHOLDER, IMAGE_SIZE


class Pix2codeOriginalModel(tf.keras.models.Model):

    def __init__(self, output_size, *args, context_length=CONTEXT_LENGTH, activation='relu', kernel_shape=3,
                 dropout_ratio=0.25, **kwargs):

        super().__init__(*args, **kwargs)
        self.context_length = context_length
        self.output_size = output_size

        self.image_model_layers = [
            Conv2D(32, kernel_shape, padding='valid', activation=activation),
            Conv2D(32, kernel_shape, padding='valid', activation=activation),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_ratio),

            Conv2D(64, kernel_shape, padding='valid', activation=activation),
            Conv2D(64, kernel_shape, padding='valid', activation=activation),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_ratio),

            Conv2D(128, kernel_shape, padding='valid', activation=activation),
            Conv2D(128, kernel_shape, padding='valid', activation=activation),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(dropout_ratio),

            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.3),
            Dense(1024, activation='relu'),
            Dropout(0.3),

            RepeatVector(context_length)
        ]
        self.language_model_layers = [
            tf.keras.layers.LSTM(128, return_sequences=True),  # , input_shape=(CONTEXT_LENGTH, output_size)))
            tf.keras.layers.LSTM(128, return_sequences=True)
        ]

        self.decoder_layers = [
            tf.keras.layers.LSTM(512, return_sequences=True),
            tf.keras.layers.LSTM(512, return_sequences=False),
            Dense(output_size, activation='softmax')
        ]

    def call(self, inputs):
        img_inp = inputs['img_data']
        context_inp = inputs['context']

        for layer in self.image_model_layers:
            img_inp = layer(img_inp)

        for layer in self.language_model_layers:
            context_inp = layer(context_inp)

        decoder = concatenate([img_inp, context_inp])
        for layer in self.decoder_layers:
            decoder = layer(decoder)
        return {'code': decoder}

    def compile(self, loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, clipvalue=1.0), **kwargs):
        # Fix the bug in tensorflow 1.15 that sets the outputnames wrong when using dicts and generators
        self.output_names = ['code']
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def predict_image(self, image, voc, img_size=IMAGE_SIZE, context_length=CONTEXT_LENGTH):

        if isinstance(image, str):
            img = preprocess_image(image, img_size)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError("Unknown handling of image input of type {}".format(type(image)))

        img_inp = np.expand_dims(img, 0)

        current_context = [voc.word2one_hot_dict[PLACEHOLDER]] * (context_length - 1)
        current_context.append(voc.word2one_hot_dict[START_WORD])

        predictions = [START_WORD]
        predicted_probas = []
        max_sequence_length = 150

        for i in range(0, max_sequence_length):
            probas = self.predict({'img_data': img_inp, 'context': np.array([current_context])})
            prediction = np.argmax(probas)
            predicted_probas.append(probas)

            # Update current context
            current_context.pop(0)
            current_context.append(tf.one_hot(prediction, self.output_size))

            word_prediction = voc.token2word_dict[prediction]
            predictions.append(word_prediction)

            if word_prediction == END_WORD:
                break
        return " ".join(predictions).replace(START_WORD + " ", "").replace(" "+END_WORD, "")
