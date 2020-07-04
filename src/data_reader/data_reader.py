# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20 20:42
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/6/28 13:57
"""

import logging
from tqdm import tqdm

from src.utils import log_title, read_json_lines, convert_list

logger = logging.getLogger(__name__)


class DataReader:
    def __init__(self, config):
        self.config = config

    def _read_data(self, data_file):
        topic = []
        triple = []
        src = []
        tgt = []

        data_iter = tqdm(list(read_json_lines(data_file)))
        for index, line in enumerate(data_iter):
            topic_seq = ' {} '.format(self.config.sep).join(line['topic'])
            triple_seq = ' {} '.format(self.config.sep).join([' '.join(v) for v in line['triples']])
            src_seq = ' {} '.format(self.config.sep).join(line['src'])
            tgt_seq = line['tgt']

            if self.config.to_lower:
                topic_seq = topic_seq.lower()
                triple_seq = triple_seq.lower()
                src_seq = src_seq.lower()
                tgt_seq = tgt_seq.lower()

            topic_tokens = [self.config.sos] + topic_seq.split() + [self.config.eos]
            triple_tokens = [self.config.sos] + triple_seq.split()[:self.config.max_triple_length] + [self.config.eos]
            src_tokens = [self.config.sos] + src_seq.split()[-self.config.max_seq_length:] + [self.config.eos]
            tgt_tokens = [self.config.sos] + tgt_seq.split()[:self.config.max_seq_length] + [self.config.eos]

            topic_ids = convert_list(topic_tokens, self.config.word_2_id, self.config.pad_id, self.config.unk_id)
            triple_ids = convert_list(triple_tokens, self.config.word_2_id, self.config.pad_id, self.config.unk_id)
            src_ids = convert_list(src_tokens, self.config.word_2_id, self.config.pad_id, self.config.unk_id)
            tgt_ids = convert_list(tgt_tokens, self.config.word_2_id, self.config.pad_id, self.config.unk_id)

            topic.append(topic_ids)
            triple.append(triple_ids)
            src.append(src_ids)
            tgt.append(tgt_ids)

            if index < 5:
                logger.info(log_title('Examples'))
                logger.info('topic tokens: {}'.format(topic_tokens))
                logger.info('topic ids: {}'.format(topic_ids))
                logger.info('triple tokens: {}'.format(triple_tokens))
                logger.info('triple ids: {}'.format(triple_ids))
                logger.info('source tokens: {}'.format(src_tokens))
                logger.info('source ids: {}'.format(src_ids))
                logger.info('target tokens: {}'.format(tgt_tokens))
                logger.info('target ids: {}'.format(tgt_ids))

        return topic, triple, src, tgt

    def read_train_data(self):
        return self._read_data(self.config.train_data)

    def read_valid_data(self):
        return self._read_data(self.config.valid_data)

    def read_test_data(self):
        return self._read_data(self.config.test_data)
