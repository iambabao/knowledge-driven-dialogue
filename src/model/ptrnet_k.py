# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/6/30 16:06
@Desc               : 
@Last modified by   : Bao
@Last modified date : 2020/6/30 16:06
"""

import tensorflow as tf

from .module.pointer_wrapper import PointerWrapper
from .module.learning_schedule import CustomSchedule
from .module.nn_utils import attention_layer, get_sparse_cross_entropy_loss, get_accuracy


class PtrNetK:
    def __init__(self, config, word_embedding_matrix):
        self.sos_id = config.sos_id
        self.eos_id = config.eos_id
        self.vocab_size = config.vocab_size
        self.oov_vocab_size = config.oov_vocab_size
        self.max_seq_length = config.max_seq_length
        self.max_triple_length = config.max_triple_length
        self.beam_size = config.top_k
        self.beam_search = config.beam_search

        self.word_em_size = config.word_em_size
        self.hidden_size = config.hidden_size
        self.attention_size = config.attention_size
        self.lr = config.lr
        self.dropout = config.dropout

        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.topic = tf.placeholder(tf.int32, [None, None], name='topic')
        self.topic_len = tf.placeholder(tf.int32, [None], name='topic_len')
        self.triple = tf.placeholder(tf.int32, [None, None], name='triple')
        self.triple_len = tf.placeholder(tf.int32, [None], name='triple_len')
        self.src = tf.placeholder(tf.int32, [None, None], name='src')
        self.src_len = tf.placeholder(tf.int32, [None], name='src_len')
        self.tgt = tf.placeholder(tf.int32, [None, None], name='tgt')
        self.tgt_len = tf.placeholder(tf.int32, [None], name='tgt_len')
        self.training = tf.placeholder(tf.bool, [], name='training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if word_embedding_matrix is not None:
            self.word_embedding = tf.keras.layers.Embedding(
                self.vocab_size + self.oov_vocab_size,
                self.word_em_size,
                embeddings_initializer=tf.constant_initializer(word_embedding_matrix),
                name='word_embedding'
            )
        else:
            self.word_embedding = tf.keras.layers.Embedding(
                self.vocab_size + self.oov_vocab_size,
                self.word_em_size,
                name='word_embedding'
            )
        self.embedding_dropout = tf.keras.layers.Dropout(self.dropout)
        self.knowledge_encoder_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.knowledge_encoder_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.seq_encoder_fw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.seq_encoder_bw = tf.nn.rnn_cell.GRUCell(self.hidden_size)
        self.decoder_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

        if config.optimizer == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr)
        elif config.optimizer == 'Adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr)
        elif config.optimizer == 'Adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.lr)
        elif config.optimizer == 'SGD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
        elif config.optimizer == 'custom':
            self.lr = CustomSchedule(self.hidden_size, self.global_step)
            self.optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9)
        else:
            assert False

        logits, self.predicted_ids = self.forward()
        self.loss = get_sparse_cross_entropy_loss(self.tgt[:, 1:], logits, self.tgt_len - 1)
        self.accu = get_accuracy(self.tgt[:, 1:], logits, self.tgt_len - 1)
        self.gradients, self.train_op = self.get_train_op()

        tf.summary.scalar('learning_rate', self.lr() if callable(self.lr) else self.lr)
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accu)
        self.summary = tf.summary.merge_all()

    def forward(self):
        # embedding
        topic_em, triple_em, src_em = self.input_embedding_layer()
        tgt_em = self.output_embedding_layer()

        # encoding
        knowledge_vec = self.knowledge_encoding_layer(topic_em, self.topic_len, src_em, self.src_len)
        enc_output, enc_state = self.sequence_encoding_layer(triple_em, self.triple_len)

        # decoding in training
        logits = self.training_decoding_layer(
            enc_output, self.triple, self.triple_len, knowledge_vec, enc_state, tgt_em[:, :-1], self.tgt_len - 1
        )

        # decoding in testing
        if not self.beam_search:
            predicted_ids = self.inference_decoding_layer(
                enc_output, self.triple, self.triple_len, knowledge_vec, enc_state, beam_search=self.beam_search
            )
        else:
            # tiled to beam size
            tiled_enc_output = tf.contrib.seq2seq.tile_batch(enc_output, multiplier=self.beam_size)
            tiled_triple = tf.contrib.seq2seq.tile_batch(self.triple, multiplier=self.beam_size)
            tiled_triple_len = tf.contrib.seq2seq.tile_batch(self.triple_len, multiplier=self.beam_size)
            tiled_knowledge_vec = tf.contrib.seq2seq.tile_batch(knowledge_vec, multiplier=self.beam_size)
            tiled_enc_state = tf.contrib.seq2seq.tile_batch(enc_state, multiplier=self.beam_size)
            predicted_ids = self.inference_decoding_layer(
                tiled_enc_output, tiled_triple, tiled_triple_len,
                tiled_knowledge_vec, tiled_enc_state, beam_search=self.beam_search
            )

        return logits, predicted_ids

    def get_train_op(self):
        gradients = tf.gradients(self.loss, tf.trainable_variables())
        gradients, _ = tf.clip_by_global_norm(gradients, 5)
        train_op = self.optimizer.apply_gradients(zip(gradients, tf.trainable_variables()), self.global_step)

        return gradients, train_op

    def input_embedding_layer(self):
        with tf.device('/cpu:0'):
            topic_em = self.word_embedding(self.topic)
            triple_em = self.word_embedding(self.triple)
            src_em = self.word_embedding(self.src)
        topic_em = self.embedding_dropout(topic_em, training=self.training)
        triple_em = self.embedding_dropout(triple_em, training=self.training)
        src_em = self.embedding_dropout(src_em, training=self.training)

        return topic_em, triple_em, src_em

    def output_embedding_layer(self):
        with tf.device('/cpu:0'):
            tgt_em = self.word_embedding(self.tgt)
        tgt_em = self.embedding_dropout(tgt_em, training=self.training)

        return tgt_em

    def knowledge_encoding_layer(self, topic_em, topic_len, src_em, src_len):
        with tf.variable_scope('knowledge_encoding_layer'):
            _, topic_enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.knowledge_encoder_fw,
                self.knowledge_encoder_bw,
                topic_em,
                topic_len,
                dtype=tf.float32
            )
            topic_enc_state = tf.concat(topic_enc_state, axis=-1)

            src_enc_output, _ = tf.nn.bidirectional_dynamic_rnn(
                self.knowledge_encoder_fw,
                self.knowledge_encoder_bw,
                src_em,
                src_len,
                dtype=tf.float32
            )
            src_enc_output = tf.concat(src_enc_output, axis=-1)

            mask = tf.sequence_mask(src_len)
            mask = tf.cast(tf.logical_not(mask), dtype=tf.float32)
            knowledge_vec = attention_layer(topic_enc_state, src_enc_output, src_enc_output, mask)

        return knowledge_vec

    def sequence_encoding_layer(self, triple_em, triple_len):
        with tf.variable_scope('sequence_encoding_layer'):
            enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.seq_encoder_fw,
                self.seq_encoder_bw,
                triple_em,
                triple_len,
                dtype=tf.float32
            )
            enc_output = tf.concat(enc_output, axis=-1)
            enc_state = enc_state[0] + enc_state[1]

        return enc_output, enc_state

    def training_decoding_layer(self, memory, memory_ids, memory_len, knowledge_vec, cell_state, tgt_em, tgt_len):
        with tf.variable_scope('decoder', reuse=False):
            # add dual pointer mechanism to decoder cell
            decoder_cell = PointerWrapper(
                self.decoder_cell,
                self.attention_size,
                memory,
                memory_ids,
                memory_len,
                knowledge_vec,
                gen_vocab_size=self.vocab_size,
                whole_vocab_size=self.vocab_size + self.oov_vocab_size
            )

            dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(memory)[0], dtype=tf.float32)
            dec_initial_state = dec_initial_state.clone(cell_state=cell_state)

            # build teacher forcing decoder
            helper = tf.contrib.seq2seq.TrainingHelper(tgt_em, tgt_len)
            decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state)

            # decoding
            final_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, impute_finished=True)

            logits = final_outputs.rnn_output

        return logits

    def inference_decoding_layer(self, memory, memory_ids, memory_len, knowledge_vec, cell_state, beam_search):
        with tf.variable_scope('decoder', reuse=True):
            # add dual pointer mechanism to decoder cell
            decoder_cell = PointerWrapper(
                self.decoder_cell,
                self.attention_size,
                memory,
                memory_ids,
                memory_len,
                knowledge_vec,
                gen_vocab_size=self.vocab_size,
                whole_vocab_size=self.vocab_size + self.oov_vocab_size
            )

            dec_initial_state = decoder_cell.zero_state(batch_size=tf.shape(memory)[0], dtype=tf.float32)
            dec_initial_state = dec_initial_state.clone(cell_state=cell_state)

            if not beam_search:
                # build greedy decoder
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    self.word_embedding,
                    tf.fill([self.batch_size], self.sos_id),
                    self.eos_id
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, dec_initial_state)
            else:
                # build beam search decoder
                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=decoder_cell,
                    embedding=self.word_embedding,
                    start_tokens=tf.fill([self.batch_size], self.sos_id),
                    end_token=self.eos_id,
                    initial_state=dec_initial_state,
                    beam_width=self.beam_size,
                    length_penalty_weight=0.0,
                    coverage_penalty_weight=0.0
                )

            # decoding
            final_outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=self.max_seq_length // 2)

            if not beam_search:
                predicted_ids = final_outputs.sample_id
            else:
                predicted_ids = final_outputs.predicted_ids  # (batch_size, seq_len, beam_size)
                predicted_ids = tf.transpose(predicted_ids, perm=[0, 2, 1])  # (batch_size, beam_size, seq_len)
                predicted_ids = predicted_ids[:, 0, :]  # keep top one

        return predicted_ids
