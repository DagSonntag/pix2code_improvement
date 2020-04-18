import tensorflow as tf

from tensorflow.python.keras.layers import Input, Dense, Dropout, \
                         RepeatVector, concatenate, \
                         Conv2D, MaxPooling2D, Flatten

from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Sequential
from constants import CONTEXT_LENGTH


class Pix2codeOrgModel(tf.keras.models.Model):

    def __init__(self, input_shape, output_size):

        image_model = tf.keras.Sequential()
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Flatten())
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))

        image_model.add(RepeatVector(CONTEXT_LENGTH))

        visual_input = Input(shape=input_shape)
        encoded_image = image_model(visual_input)

        language_model = Sequential()
        language_model.add(tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(tf.keras.layers.LSTM(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([encoded_image, encoded_text])

        decoder = tf.keras.layers.LSTM(512, return_sequences=True)(decoder)
        decoder = tf.keras.layers.LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        super().__init__(inputs=[visual_input, textual_input], outputs=decoder)

        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)

        self.compile(loss='categorical_crossentropy', optimizer=optimizer)


