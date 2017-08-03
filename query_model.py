
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from .utils import lazy_property, weight_and_bias, DTYPE


class QueryModel(object):

    def __init__(self, data, target, num_hidden=200, num_layers=3, dropout=0.5):
        self.data = data
        self.target = target
        self.dropout = dropout
        self._keep_prob = 1 - dropout
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self.prediction
        self.optimize
        self.error    

    @lazy_property
    def prediction(self):
        """
        Create predictions from the calculated logits.
        """
        prediction = tf.nn.softmax(self._logits)
        return prediction

    @lazy_property
    def cost(self):
        """
        Calculate the loss for a batch from its calculated logits.
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._logits, labels=self.target)
        # Mask out error from output beyond each sequence length
        cross_entropy *= self._sequence_mask
        # Average over actual sequence lengths
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, DTYPE)
        return tf.reduce_mean(cross_entropy)

    @lazy_property
    def optimize(self):
        learning_rate = 0.003
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.minimize(self.cost)

    @lazy_property
    def error(self):
        mistakes = tf.not_equal(tf.arg_max(self.target, 2), tf.argmax(self.prediction, 2))
        mistakes = tf.cast(mistakes, DTYPE)
        # Mask out mistakes outside of each sequence
        mistakes *= self._sequence_mask
        # Average over actual sequence lengths
        mistakes = tf.reduce_sum(mistakes, reduction_indices=2)
        mistakes /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(mistakes)

    @lazy_property
    def output(self):
        o, _ = self._base_network()
        return o

    @lazy_property
    def state(self):
        _, s = self._base_network()
        return s

    @lazy_property
    def _sequence_mask(self):
        return tf.sign(tf.reduce_max(tf.abs(self.target), reduction_indices=2))

    @lazy_property
    def _logits(self):
        """
        Calculate logits for sequences processed by the base RNN.
        """
        # Get outputs and states from base RNN
        output, state = self._base_network()
        # Softmax layer
        max_length = int(self.target.get_shape()[1])
        num_classes = int(self.target.get_shape()[2])
        weight, bias = weight_and_bias(self._num_hidden, num_classes)
        # Flatten to apply same weights to all time steps
        output = tf.reshape(output, [-1, self._num_hidden])
        logits = tf.matmul(output, weight) + bias
        logits = tf.reshape(logits, [-1, max_length, num_classes])
        return logits

    @lazy_property
    def _base_network(self):
        """
        Form an RNN for sequence labeling capable of handling variable length sequences.
        """
        # Recurrent network
        cells = []
        for _ in range(self._num_layers):
            cell = tf.nn.rnn_cell.GRUCell(self._num_hidden)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._keep_prob)
            cells.append(cell)
        network = tf.nn.rnn_cell.MultiRNNCell(network)
        return tf.nn.dynamic_rnn(
            network,
            self.data,
            dtype=DTYPE,
            sequence_length=self.length
        )

    @lazy_property
    def length(self):
        """
        Take a [batch_size, max_sequence_length, data_dim] size tensor and turn into
        a 1D tensor of length [batch_size] where each element is the length of its
        respective example from the batch.

        For dynamic sequence lengths to work no pieces of data may be encoded as the
        zero vector because it won't contribute to the calculated length of the sequence.
        """
        used = tf.sign(tf.reduce_max(tf.abs(self.data), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
    
    