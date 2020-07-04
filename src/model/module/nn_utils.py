# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20 20:42
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/6/28 13:57
"""

import tensorflow as tf


def expand_attention_score(score, memory_ids, whole_vocab_size):
    """

    :param score: (batch_size, seq_length)
    :param memory_ids: (batch_size, seq_length)
    :param whole_vocab_size:
    :return:
    """
    batch_size = tf.shape(memory_ids)[0]
    sequence_length = tf.shape(memory_ids)[1]

    row_indices = tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1]), [1, sequence_length, 1])
    col_indices = tf.expand_dims(memory_ids, axis=-1)
    indices = tf.reshape(tf.concat([row_indices, col_indices], axis=-1), [-1, 2])  # (batch_size * seq_length, 2)
    values = tf.reshape(score, [-1])  # (batch_size * seq_length)

    expanded_score = tf.scatter_nd(indices, values, [batch_size, whole_vocab_size])

    return expanded_score


def bahdanau_attention(h, s, v, attention_vec, h_length=None):
    """

    :param h: (batch_size, sequence_length, attention_size)
    :param s: (batch_size, attention_size)
    :param v: (batch_size, attention_size)
    :param attention_vec: (attention_size,)
    :param h_length: (batch_size,)
    :return:
    """
    s = tf.expand_dims(s, axis=1)  # (batch_size, 1, attention_size)
    v = tf.expand_dims(v, axis=1)  # (batch_size, 1, attention_size)
    attention_score = tf.reduce_sum(attention_vec * tf.math.tanh(h + s + v), axis=-1)

    if h_length is not None:
        mask = tf.sequence_mask(h_length)
        mask = tf.cast(tf.logical_not(mask), dtype=tf.float32)
        attention_score += -1e9 * mask

    attention_score = tf.math.softmax(attention_score, axis=-1)  # (batch_size, seq_length)

    return attention_score


def attention_layer(query, key, value, mask):
    """

    :param query: (batch_size, attention_size)
    :param key: (batch_size, sequence_length, attention_size)
    :param value: (batch_size, sequence_length, hidden_size)
    :param mask: (batch_size, sequence_length)
    :return:
    """
    attention = tf.reduce_sum(tf.expand_dims(query, axis=1) * key, axis=-1)
    attention += -1e9 * mask
    attention = tf.math.softmax(attention, axis=-1)

    context_vec = tf.reduce_sum(tf.expand_dims(attention, axis=-1) * value, axis=1)

    return context_vec


def get_sparse_softmax_cross_entropy_loss(labels, logits, mask_sequence_length=None):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if mask_sequence_length is not None:
        mask = tf.sequence_mask(mask_sequence_length, dtype=tf.float32)
        loss = tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, axis=-1))
    else:
        loss = tf.reduce_mean(loss)

    return loss


def get_sparse_cross_entropy_loss(y_true, y_pred, mask_sequence_length=None):
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    if mask_sequence_length is not None:
        mask = tf.sequence_mask(mask_sequence_length, dtype=tf.float32)
        loss = tf.reduce_mean(tf.reduce_sum(mask * loss, axis=-1) / tf.reduce_sum(mask, axis=-1))
    else:
        loss = tf.reduce_mean(loss)

    return loss


def get_accuracy(y_true, y_pred, mask_sequence_length=None):
    pred_ids = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
    accuracy = tf.cast(tf.equal(y_true, pred_ids), tf.float32)
    if mask_sequence_length is not None:
        mask = tf.sequence_mask(mask_sequence_length, dtype=tf.float32)
        accuracy = tf.reduce_mean(tf.reduce_sum(mask * accuracy, axis=-1) / tf.reduce_sum(mask, axis=-1))
    else:
        accuracy = tf.reduce_mean(accuracy)

    return accuracy
