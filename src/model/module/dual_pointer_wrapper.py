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


class DualPointerWrapperState(collections.namedtuple('DualPointerWrapperState',
                                                     ('cell_state', 'time',
                                                      'alignments_a', 'alignments_b',
                                                      'coverage_a', 'coverage_b'))):
    def clone(self, **kwargs):
        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tensor_util.with_same_shape(old, new)
            return new

        return nest.map_structure(
            with_same_shape,
            self,
            super(DualPointerWrapperState, self)._replace(**kwargs)
        )


class DualPointerWrapper(tf.nn.rnn_cell.RNNCell):
    def __init__(self,
                 cell,
                 attention_size,
                 memory_a,
                 memory_a_ids,
                 memory_a_length,
                 memory_b,
                 memory_b_ids,
                 memory_b_length,
                 knowledge_vec,
                 gen_vocab_size,
                 whole_vocab_size=None,
                 initial_cell_state=None,
                 name=None):
        """
        Args:
            cell: RNN cell
            attention_size:
            memory_a:
            memory_a_ids:
            memory_a_length:
            memory_b:
            memory_b_ids:
            memory_b_length:
            knowledge_vec:
            gen_vocab_size:
            whole_vocab_size:
            initial_cell_state:
            name:
        """
        super(DualPointerWrapper, self).__init__(name=name)
        self._cell = cell
        self._attention_size = attention_size
        self._memory_a = memory_a
        self._memory_a_ids = memory_a_ids
        self._memory_a_length = memory_a_length
        self._memory_b = memory_b
        self._memory_b_ids = memory_b_ids
        self._memory_b_length = memory_b_length
        self._knowledge_vec = knowledge_vec
        self._gen_vocab_size = gen_vocab_size
        self._whole_vocab_size = whole_vocab_size or gen_vocab_size
        if self._whole_vocab_size < self._gen_vocab_size:
            raise ValueError(
                'whole_vocab_size must greater or equal to gen_vocab_size. {} vs. {}'
                    .format(self._whole_vocab_size, self._gen_vocab_size)
            )
        self._initial_cell_state = initial_cell_state

        self._dense_h_a = tf.layers.Dense(self._attention_size, use_bias=False, name='dense_h_a')
        self._dense_h_b = tf.layers.Dense(self._attention_size, use_bias=False, name='dense_h_b')
        self._dense_s = tf.layers.Dense(self._attention_size, use_bias=False, name='dense_s')
        self._dense_v = tf.layers.Dense(self._attention_size, use_bias=False, name='dense_v')
        self._attention_vec_a = tf.get_variable('attention_vec_a', [self._attention_size])
        self._attention_vec_b = tf.get_variable('attention_vec_b', [self._attention_size])
        self._dense_prob = tf.layers.Dense(3, use_bias=False, name='dense_prob')
        self._output_layer_1 = tf.layers.Dense(self._attention_size, name='output_layer_1')
        self._output_layer_2 = tf.layers.Dense(self._gen_vocab_size, name='output_layer_2')

    def __call__(self, inputs, state, scope=None):
        if not isinstance(state, DualPointerWrapperState):
            raise TypeError(
                'Expected state to be instance of DualPointerWrapperState. Received type {} instead.'.format(type(state))
            )
        prev_cell_state = state.cell_state
        prev_time = state.time
        prev_coverage_a = state.coverage_a
        prev_coverage_b = state.coverage_b

        cell_outputs, cell_state = self._cell(inputs, prev_cell_state, scope)

        h_a = self._dense_h_a(self._memory_a)  # (batch_size, seq_length, attention_size)
        h_b = self._dense_h_b(self._memory_b)  # (batch_size, seq_length, attention_size)
        s = self._dense_s(cell_outputs)  # (batch_size, attention_size)
        v = self._dense_v(self._knowledge_vec)  # (batch_size, attention_size)
        attention_a_score = bahdanau_attention(h_a, s, v, self._attention_vec_a, self._memory_a_length)
        attention_b_score = bahdanau_attention(h_b, s, v, self._attention_vec_b, self._memory_b_length)

        context_a = tf.reduce_sum(tf.expand_dims(attention_a_score, axis=-1) * self._memory_a, axis=1)
        context_b = tf.reduce_sum(tf.expand_dims(attention_b_score, axis=-1) * self._memory_b, axis=1)
        context = tf.concat([cell_outputs, context_a, context_b, self._knowledge_vec], axis=-1)
        generate_score = tf.math.softmax(self._output_layer_2(tf.math.tanh(self._output_layer_1(context))), axis=-1)

        expanded_g_score = tf.pad(generate_score, [[0, 0], [0, self._whole_vocab_size - self._gen_vocab_size]])
        expanded_a_score = expand_attention_score(attention_a_score, self._memory_a_ids, self._whole_vocab_size)
        expanded_b_score = expand_attention_score(attention_b_score, self._memory_b_ids, self._whole_vocab_size)
        all_score = tf.stack([expanded_g_score, expanded_a_score, expanded_b_score], axis=1)  # (batch_size, 3, whole_vocab_size)

        prob = tf.math.softmax(self._dense_prob(context), axis=-1)
        outputs = tf.reduce_sum(tf.expand_dims(prob, axis=-1) * all_score, axis=1)

        alignments_a = attention_a_score
        alignments_b = attention_b_score
        coverage_a =  prev_coverage_a + attention_a_score
        coverage_b =  prev_coverage_b + attention_b_score
        state = DualPointerWrapperState(cell_state=cell_state, time=prev_time + 1,
                                        alignments_a=alignments_a, alignments_b=alignments_b,
                                        coverage_a=coverage_a, coverage_b=coverage_b)
        return outputs, state

    @property
    def state_size(self):
        """
            size(s) of state(s) used by this cell.

            It can be represented by an Integer, a TensorShape or a tuple of Integers
            or TensorShapes.
        """
        return DualPointerWrapperState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            alignments_a=self._memory_a_ids.shape[1].value,
            alignments_b=self._memory_a_ids.shape[1].value,
            coverage_a=self._memory_a_ids.shape[1].value,
            coverage_b=self._memory_b_ids.shape[1].value
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
            alignments_a = tf.zeros([batch_size, tf.shape(self._memory_a_ids)[1]], tf.float32)
            alignments_b = tf.zeros([batch_size, tf.shape(self._memory_a_ids)[1]], tf.float32)
            coverage_a = tf.zeros([batch_size, tf.shape(self._memory_a_ids)[1]], tf.float32)
            coverage_b = tf.zeros([batch_size, tf.shape(self._memory_b_ids)[1]], tf.float32)
            return DualPointerWrapperState(cell_state=cell_state, time=time,
                                           alignments_a=alignments_a, alignments_b=alignments_b,
                                           coverage_a=coverage_a, coverage_b=coverage_b)
