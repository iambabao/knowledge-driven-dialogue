# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/6/30 16:06
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/6/30 16:06
"""

import collections

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.framework.python.framework import tensor_util

from .nn_utils import bahdanau_attention, expand_attention_score


class PointerWrapperState(collections.namedtuple('PointerWrapperState',
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
            super(PointerWrapperState, self)._replace(**kwargs)
        )


class PointerWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 cell,
                 attention_size,
                 memory,
                 memory_ids,
                 memory_length,
                 knowledge_vec,
                 gen_vocab_size,
                 whole_vocab_size=None,
                 initial_cell_state=None,
                 name=None):
        """
        Args:
            cell: RNN cell
            attention_size:
            memory:
            memory_ids:
            memory_length:
            knowledge_vec:
            gen_vocab_size:
            whole_vocab_size:
            initial_cell_state:
            name:
        """
        super(PointerWrapper, self).__init__(name=name)
        self._cell = cell
        self._attention_size = attention_size
        self._memory = memory
        self._memory_ids = memory_ids
        self._memory_length = memory_length
        self._knowledge_vec = knowledge_vec
        self._gen_vocab_size = gen_vocab_size
        self._whole_vocab_size = whole_vocab_size or gen_vocab_size
        if self._whole_vocab_size < self._gen_vocab_size:
            raise ValueError(
                'whole_vocab_size must greater or equal to gen_vocab_size. {} vs. {}'
                    .format(self._whole_vocab_size, self._gen_vocab_size)
            )
        self._initial_cell_state = initial_cell_state

        self._dense_h = tf.layers.Dense(self._attention_size, use_bias=False, name='dense_h')
        self._dense_s = tf.layers.Dense(self._attention_size, use_bias=False, name='dense_s')
        self._dense_v = tf.layers.Dense(self._attention_size, use_bias=False, name='dense_v')
        self._attention_vec = tf.get_variable('attention_vec', [self._attention_size])
        self._dense_prob = tf.layers.Dense(1, use_bias=False, name='dense_prob')
        self._output_layer_1 = tf.layers.Dense(self._attention_size, name='output_layer_1')
        self._output_layer_2 = tf.layers.Dense(self._gen_vocab_size, name='output_layer_2')

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, PointerWrapperState):
            raise TypeError(
                'Expected state to be instance of PointerWrapperState. Received type {} instead.'.format(type(state))
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
        generate_score = tf.math.softmax(self._output_layer_2(tf.math.tanh(self._output_layer_1(context))), axis=-1)

        expanded_g_score = tf.pad(generate_score, [[0, 0], [0, self._whole_vocab_size - self._gen_vocab_size]])
        expanded_c_score = expand_attention_score(attention_score, self._memory_ids, self._whole_vocab_size)

        prob_gen = tf.math.sigmoid(self._dense_prob(context))
        outputs = prob_gen * expanded_g_score + (1 - prob_gen) * expanded_c_score

        alignments = attention_score
        coverage =  prev_coverage + attention_score
        state = PointerWrapperState(cell_state=cell_state, time=prev_time + 1,
                                    alignments=alignments, coverage=coverage)
        return outputs, state

    @property
    def state_size(self):
        """
            size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return PointerWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            alignments=self._memory_ids.shape[1].value,
            coverage=self._memory_ids.shape[1].value
        )

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self._whole_vocab_size

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + 'ZeroState', values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            time = tf.zeros([], tf.int32)
            alignments = tf.zeros([batch_size, tf.shape(self._memory_ids)[1]], tf.float32)
            coverage = tf.zeros([batch_size, tf.shape(self._memory_ids)[1]], tf.float32)
            return PointerWrapperState(cell_state=cell_state, time=time,
                                       alignments=alignments, coverage=coverage)
