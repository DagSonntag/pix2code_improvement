#from tensorflow_addons.seq2seq.sampler import Sampler, _unstack_ta

import tensorflow as tf
import tensorflow_addons as tfa
#from tensorflow_addons.seq2seq import decoder
from typeguard import typechecked

_transpose_batch_time = tfa.seq2seq.decoder._transpose_batch_time


class TrainingInferenceSampler(tfa.seq2seq.sampler.Sampler):
    """
    A custom sampler that runs both training (serving inputs as next outputs) and inference (uses previous output as
    input). Uses tf.one_hot(tf.argmax(prev_output)) as next input in inference mode.
    Returned sample_ids are the argmax of the RNN output logits.
    """

    def __init__(self, training, voc_size, max_sequence_length=None, embedder=None, img_input=None):
        """
        :param training: Whether the sampler should be in training mode or not (i.e. inference mode)
        :param voc_size: The size of the vocabulary, used to create the one hot encoding in inference
        :param max_sequence_length: The max sequence length used in inference mode
        """
        self._batch_size = None

        self.training = training
        self.voc_size = voc_size
        self.max_sequence_length = max_sequence_length
        self.sequence_length = tf.fill(
            [0], self.max_sequence_length, name="sequence_length"
        )
        self.embedder = embedder
        self.img_input = img_input

    @property
    def batch_size(self):
        if self._batch_size is None:
            raise ValueError("batch_size accessed before initialize was called")
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, inputs):
        """
        Initializes the sampler. Called from the decoder on initialization.
        :param inputs: Inputs used in the training/inference. In case of inference only the first time-step is used
        :return: (finished, next_inputs), a tuple of two items. The first item is a boolean vector to indicate whether
        the item in the batch has finished. The second item is the first slide of input data based on the timestep
        dimension (usually the second dim of the input).
        """
        self.inputs = tf.convert_to_tensor(inputs, name="inputs")

        inputs = tf.nest.map_structure(_transpose_batch_time, inputs)
        self._batch_size = tf.shape(tf.nest.flatten(inputs)[0])[1]
        self.input_tas = tf.nest.map_structure(tfa.seq2seq.sampler._unstack_ta, inputs)
        if self.training:
            # As the input tensor has been converted to time major,
            # the maximum sequence length should be inferred from
            # the first dimension.
            max_seq_len = tf.shape(tf.nest.flatten(inputs)[0])[0]
            self.sequence_length = tf.fill(
                [self.batch_size], max_seq_len, name="sequence_length"
            )
        else:
            self.sequence_length = tf.fill(
                [self.batch_size], self.max_sequence_length, name="sequence_length"
            )

        self.zero_inputs = tf.nest.map_structure(
            lambda inp: tf.zeros_like(inp[0, :]), inputs
        )

        finished = tf.equal(0, self.sequence_length)
        all_finished = tf.reduce_all(finished)
        next_inputs = tf.cond(
            all_finished,
            lambda: self.zero_inputs,
            lambda: tf.nest.map_structure(lambda inp: inp.read(0), self.input_tas),
        )

        if self.img_input is not None:
            next_inputs = tf.concat([next_inputs, self.img_input], axis=-1)

        return finished, next_inputs

    def sample(self, time, outputs, state):
        del state
        sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids):
        """
        Calculates the next input for each timepoint. For this sampler this is in training mode the next datapoint in
        the input (stored), while in inference mode the output
        :param time: The time-index, adds one each step (recursion) (int)
        :param outputs: The outputs from the decoder-cell
        :param state: The output-state of the decoder-cell
        :param sample_ids: The output of the sample method above
        :return: (finished, next_inputs, state) where the finished is true if the sequence length has been reached,
        the next inputs  set as described above, and the next state is just the state in the method input (not altered)
        """
        next_time = time + 1
        finished = next_time >= self.sequence_length
        all_finished = tf.reduce_all(finished)

        def read_from_ta(inp):
            return inp.read(next_time)

        if self.training:
            next_inputs = tf.cond(
                all_finished,
                lambda: self.zero_inputs,
                lambda: tf.nest.map_structure(read_from_ta, self.input_tas),
            )
        else:
            next_inputs = tf.cond(
                all_finished,
                lambda: self.zero_inputs,
                lambda: tf.one_hot(sample_ids, self.voc_size, dtype=tf.float32) if self.embedder is None else self.embedder(sample_ids),
            )
        if self.img_input is not None:
            next_inputs = tf.concat([next_inputs, self.img_input], axis=-1)

        return finished, next_inputs, state
