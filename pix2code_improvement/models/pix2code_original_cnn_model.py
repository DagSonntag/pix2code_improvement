import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop


class Pix2CodeOriginalCnnModel(tf.keras.Model):
    def __init__(self, output_names, dropout_ratio=0.25, activation='relu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_length = len(output_names)
        self.layer_output_names = output_names
        self.layer_list = []
        # Input = 256
        self.layer_list.append(
            Conv2D(32, (3, 3), padding='valid', name='cnn_256pix_1', activation=activation, strides=1))
        self.layer_list.append(
            Conv2D(32, (3, 3), padding='valid', name='cnn_256pix_2', activation=activation, strides=1))

        self.layer_list.append(MaxPooling2D(pool_size=(2, 2)))
        self.layer_list.append(Dropout(dropout_ratio))
        # 128
        self.layer_list.append(
            Conv2D(64, (3, 3), padding='valid', name='cnn_128pix_1', activation=activation, strides=1))
        self.layer_list.append(
            Conv2D(64, (3, 3), padding='valid', name='cnn_128pix_2', activation=activation, strides=1))

        self.layer_list.append(MaxPooling2D(pool_size=(2, 2)))
        self.layer_list.append(Dropout(dropout_ratio))
        # 64
        self.layer_list.append(Conv2D(128, (3, 3), padding='valid', name='cnn_64pix_1', activation=activation))
        self.layer_list.append(Conv2D(128, (3, 3), padding='valid', name='cnn_64pix_2', activation=activation))
        self.layer_list.append(MaxPooling2D(pool_size=(2, 2)))
        self.layer_list.append(Dropout(dropout_ratio))

        self.layer_list.append(Flatten())
        self.layer_list.append(Dense(1024, activation=activation))
        self.layer_list.append(Dropout(0.3))
        self.layer_list.append(Dense(1024, activation=activation))
        self.layer_list.append(Dropout(0.3))
        self.layer_list.append(
            [Dense(1, name=output_names[i], activation=activation)
             for i in range(self.output_length)])

    def call(self, inputs):
        inp = inputs['img_data']
        for layer in self.layer_list[:-1]:
            inp = layer(inp)
        last_layers = self.layer_list[-1]
        out = {name+"_count": layer(inp) for name, layer in zip(self.layer_output_names, last_layers)}
        return out

    def compile(self, loss='mse', optimizer=RMSprop(lr=0.0001, clipvalue=1.0), **kwargs):
        self.output_names = sorted([key + "_count" for key in self.layer_output_names])
        return super().compile(loss=loss, optimizer=optimizer, **kwargs)

    def predict(self, *args, return_as_dict=True, **kwargs):
        pred = super().predict(*args, **kwargs)
        if return_as_dict:
            return {key: val for key, val in zip(self.output_names, pred)}
        else:
            return pred

