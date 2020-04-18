import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop


class CounterUnit(tf.keras.layers.Layer):
    def __init__(self, layer_size, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.main_layers = [
            Flatten(name='flatten'),
            Dense(layer_size, activation=activation, name="counter1"),
            Dense(layer_size, activation=activation, name="counter2")]
        self.last_layer = Dense(1, activation=activation, name='counter_out')

    def call(self, inputs):
        inp = inputs
        for layer in self.main_layers:
            inp = layer(inp)
        return self.last_layer(inp)


class ShallowCnnUnit(tf.keras.layers.Layer):
    def __init__(self, image_count_words, kernel_shape=7, dropout_ratio=0.1, activation='relu', **kwargs):
        super().__init__(**kwargs)

        if isinstance(kernel_shape, tuple) or isinstance(kernel_shape, list):
            kernel_shape = tuple(kernel_shape)
        else:
            kernel_shape = (kernel_shape, kernel_shape)

        self.core_cnn_layer_list = [
            # Input = 256
            Conv2D(32, kernel_shape, padding='same', name='cnn_256pix_1', activation=activation),
            Conv2D(32, kernel_shape, padding='same', name='cnn_256pix_2', strides=2, activation=activation),
            Dropout(dropout_ratio, name='drop_cnn256'),
            # 128
            Conv2D(64, kernel_shape, padding='same', name='cnn_128pix_1', strides=1, activation=activation),
            Conv2D(64, kernel_shape, padding='same', name='cnn_128pix_2', strides=2, activation=activation),
            Dropout(dropout_ratio, name='drop_cnn138'),
            # 64
            Conv2D(128, kernel_shape, padding='same', name='cnn_64pix_1', strides=1, activation=activation),
            Dropout(dropout_ratio, name='drop_cnn64')
        ]
        # the parallel cnn layers
        self.parallel_cnn_output_layers = [Conv2D(1, kernel_shape, padding='same', name=layer_name)
                                           for layer_name in image_count_words]

    def call(self, inputs):
        inp = inputs
        for layer in self.core_cnn_layer_list:
            inp = layer(inp)
        return [layer(inp) for layer in self.parallel_cnn_output_layers]


class ShallowCnnModel(tf.keras.Model):
    """
    Model construction:
    Vars:
    n_outputs: the number of outputs (type of objects) to count
    dense_layer_size: the number of neurons in each of the dense layers
    * 1 common cnn that takes 256x256 image inputs and transforms this down to 64x64pix images.
    * One last layer consisting of multiple (n_outputs) parallel cnns. Each cnn outputs, one feature map times 64x64 pixels. This (combined) layer builds the auxilary output
    * On top of this each of these cnn-maps has (in parallel) a "counter unit" that consists of 3 dense layers on top of eachother, the first two with dense_layer_size number of neurons,
      the last with a single neuron (the output layer). Combined these output layers build up the main output (used for training)
    """

    def __init__(self, image_count_words, kernel_shape=7, dense_layer_size=256, activation='relu', dropout_ratio=0.25,
                 image_out=False, *args, **kwargs):

        super().__init__(*args, **kwargs)


        self.output_length = len(image_count_words)
        self.image_out = image_out
        self.image_count_words = image_count_words
        self.shallow_cnn_unit = ShallowCnnUnit(image_count_words=image_count_words, kernel_shape=kernel_shape,
                                               dropout_ratio=dropout_ratio, activation=activation, name='cnn_unit')
        self.parallel_counter_unit = CounterUnit(layer_size=dense_layer_size, activation=activation,
                                                 name='counter_unit')

    def call(self, inputs):
        inp = inputs['img_data']
        cnn_output_layers = self.shallow_cnn_unit(inp)
        output_layers = [self.parallel_counter_unit(img_out) for img_out in cnn_output_layers]
        output_layers = {key+"_count": layer for key, layer in zip(self.image_count_words, output_layers)}
        if self.image_out:
            output_layers.update({"img_out_" + key: layer for key, layer in zip(self.image_count_words,
                                                                                cnn_output_layers)})
        return output_layers

    def compile(self, loss='mse', optimizer=RMSprop(lr=0.0001, clipvalue=1.0), *args, **kwargs):
        # Fix the bug in tensorflow 2.1 that sets the outputnames wrong when using dicts and generators
        if self.image_out:
            names = ([key + "_count" for key in self.image_count_words]
                     + ["img_out_" + key for key in self.image_count_words])
        else:
            names = [key + "_count" for key in self.image_count_words]
        self.output_names = sorted(names)
        return super().compile(loss=loss, optimizer=optimizer, *args, **kwargs)

    def predict(self, *args, return_as_dict=True, **kwargs):
        pred = super().predict(*args, **kwargs)
        if return_as_dict:
            return {key: val for key, val in zip(self.output_names, pred)}
        else:
            return pred