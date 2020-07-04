# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20 20:42
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/6/28 13:57
"""

import collections

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util

from .nn_utils import bahdanau_attention


class AttWrapperState(collections.namedtuple('AttWrapperState',
                                             ('cell_state', 'time',
                                              'alignments', 'coverage'))):
    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(AttWrapperState, self)._replace(**kwargs)
        )


class AttWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 cell,
                 attention_size,
                 memory,
                 memory_length,
                 knowledge_vec,
                 initial_cell_state=None,
                 name=None):
        """
        Args:
            cell: RNN cell
            attention_size:
            memory:
            memory_length:
            knowledge_vec:
            initial_cell_state:
            name:
        """
        super(AttWrapper, self).__init__(name=name)
        self._cell = cell
        self._attention_size = attention_size
        self._memory = memory
        self._memory_length = memory_length
        self._knowledge_vec = knowledge_vec
        self._initial_cell_state = initial_cell_state

        self._dense_h = tf.layers.Dense(self._attention_size, use_bias=False, name='dense_h')
        self._dense_s = tf.layers.Dense(self._attention_size, use_bias=False, name='dense_s')
        self._dense_v = tf.layers.Dense(self._attention_size, use_bias=False, name='dense_v')
        self._attention_vec = tf.get_variable('attention_vec', [self._attention_size])
        self._output_layer = tf.layers.Dense(self._attention_size, name='output_layer')

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, AttWrapperState):
            raise TypeError(
                'Expected state to be instance of AttWrapperState. Received type {} instead.'.format(type(state))
            )
        prev_cell_state = state.cell_state
        prev_time = state.time
        prev_coverage = state.coverage

        cell_outputs, cell_state = self._cell(inputs, prev_cell_state, scope)

        h = self._dense_h(self._memory)  # (batch_size, seq_length, attention_size)
        s = self._dense_s(cell_outputs)  # (batch_size, attention_size)
        v = self._dense_v(self._knowledge_vec)  # (batch_size, attention_size)
        attention_score = bahdanau_attention(h, s, v, self._attention_vec, self._memory_length)

        context = tf.reduce_sum(tf.expand_dims(attention_score, axis=-1) * self._memory, axis=1)
        context = tf.concat([cell_outputs, context, self._knowledge_vec], axis=-1)
        outputs = tf.math.tanh(self._output_layer(context))

        alignments = attention_score
        coverage =  prev_coverage + attention_score
        state = AttWrapperState(cell_state=cell_state, time=prev_time + 1,
                                alignments=alignments, coverage=coverage)
        return outputs, state

    @property
    def state_size(self):
        """
            size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return AttWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            alignments=self._memory.shape[1].value,
            coverage=self._memory.shape[1].value
        )

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._memory.shape[-1].value

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            time = tf.zeros([], tf.int32)
            alignments = tf.zeros([batch_size, tf.shape(self._memory)[1]], tf.float32)
            coverage = tf.zeros([batch_size, tf.shape(self._memory)[1]], tf.float32)
            return AttWrapperState(cell_state=cell_state, time=time,
                                   alignments=alignments, coverage=coverage)
