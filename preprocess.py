# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20 20:42
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/6/28 13:57
"""

import logging
import collections
from tqdm import tqdm

from src.config import Config
from src.utils import init_logger, log_title, read_json_lines, save_json_lines, save_json_dict

logger = logging.getLogger(__name__)


def build_dict(filename, config):
    counter = collections.Counter()

    for line in tqdm(list(read_json_lines(filename))):
        goal = line['goal']
        knowledge = line['knowledge']
        conversation = line['conversation']

        topic = goal[0][1:]
        for entity in topic:
            if config.to_lower:
                entity = entity.lower()
            for token in entity.strip().split():
                counter[token] += 1

        triples = knowledge + [v for v in goal[1:] if v not in knowledge]
        for triple in triples:
            for node in triple:
                if config.to_lower:
                    node = node.lower()
                for token in node.strip().split():
                    counter[token] += 1

        for sequence in conversation:
            if config.to_lower:
                sequence = sequence.lower()
            for token in sequence.strip().split():
                counter[token] += 1

    counter[config.pad] = 1e9 - config.pad_id
    counter[config.unk] = 1e9 - config.unk_id
    counter[config.sos] = 1e9 - config.sos_id
    counter[config.eos] = 1e9 - config.eos_id
    counter[config.sep] = 1e9 - config.sep_id
    counter[config.num] = 1e9 - config.num_id
    counter[config.time] = 1e9 - config.time_id
    logger.info('number of words: {}'.format(len(counter)))

    word_dict = {}
    for word, _ in counter.most_common(config.vocab_size + config.oov_vocab_size):
        word_dict[word] = len(word_dict)

    save_json_dict(word_dict, config.vocab_dict)


def generate_data(input_file, output_file, is_test=False):
    data = []
    for line in tqdm(list(read_json_lines(input_file))):
        goal = line['goal']
        knowledge = line['knowledge']

        topic = goal[0][1:]
        triples = knowledge + [v for v in goal[1:] if v not in knowledge]
        if not is_test:
            conversation = line['conversation']
            for i in range(len(conversation)):
                src = conversation[:i]
                tgt = conversation[i]
                data.append({'src': src, 'tgt': tgt, 'topic': topic, 'triples': triples})
        else:
            src = line['history']
            tgt = line['response']
            data.append({'src': src, 'tgt': tgt, 'topic': topic, 'triples': triples})

    save_json_lines(data, output_file)


def preprocess():
    logger.setLevel(logging.INFO)
    init_logger(logging.INFO, 'temp.log.txt', 'w')

    config = Config('.', 'temp')

    logger.info('building dict...')
    build_dict('data/train.txt', config)

    logger.info('generating data...')
    generate_data('data/train.txt', config.train_data)
    generate_data('data/dev.txt', config.valid_data)
    generate_data('data/test_1.txt', config.test_data, is_test=True)


if __name__ == '__main__':
    preprocess()

    logger.info(log_title('done'))
