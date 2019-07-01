"""Variational Dropout Wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class VariationalDropoutWrapper(tf.contrib.rnn.RNNCell):
    """Add variational dropout to a RNN cell."""

    def __init__(self, cell, batch_size, input_size, recurrent_keep_prob,
                 input_keep_prob):
        self._cell = cell
        self._recurrent_keep_prob = recurrent_keep_prob
        self._input_keep_prob = input_keep_prob

        def make_mask(keep_prob, units):
            random_tensor = keep_prob
            # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
            random_tensor += tf.random_uniform(tf.stack([batch_size, units]))
            return tf.floor(random_tensor) / keep_prob

        self._recurrent_mask = make_mask(recurrent_keep_prob,
                                         self._cell.state_size[0])
        self._input_mask = self._recurrent_mask

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        dropped_inputs = inputs * self._input_mask
        dropped_state = (state[0], state[1] * self._recurrent_mask)
        new_h, new_state = self._cell(dropped_inputs, dropped_state, scope)
        return new_h, new_state
