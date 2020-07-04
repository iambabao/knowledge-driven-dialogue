# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20 20:42
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/6/30 13:57
"""

import logging
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu

from src.utils import read_json_lines

logger = logging.getLogger(__name__)


def calc_bleu(references, hypotheses):
    list_of_references = [[v] for v in references]

    bleu1 = 100 * corpus_bleu(list_of_references, hypotheses, (1., 0., 0., 0.))
    bleu2 = 100 * corpus_bleu(list_of_references, hypotheses, (0.5, 0.5, 0., 0.))
    bleu3 = 100 * corpus_bleu(list_of_references, hypotheses, (0.33, 0.33, 0.33, 0.))
    bleu4 = 100 * corpus_bleu(list_of_references, hypotheses, (0.25, 0.25, 0.25, 0.25))

    return {'BLEU 1': bleu1, 'BLEU 2': bleu2, 'BLEU 3': bleu3, 'BLEU 4': bleu4}


def calc_f1(references, hypotheses):
    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    for ref, hyp in zip(references, hypotheses):
        golden_response = ''.join(ref)
        response = ''.join(hyp)
        common = Counter(response) & Counter(golden_response)
        hit_char_total += sum(common.values())
        golden_char_total += len(golden_response)
        pred_char_total += len(response)
    p = (hit_char_total / pred_char_total) if pred_char_total != 0 else 0.
    r = (hit_char_total / golden_char_total) if golden_char_total != 0 else 0.
    f1 = (100 * 2 * p * r / (p + r)) if (p + r) != 0 else 0.
    return {'F1': f1}


def calc_distinct_ngram(hypotheses, max_ngram):
    def _calc_distinct(ngram):
        ngram_total = 0.0
        ngram_distinct_count = 0.0
        counter = Counter()
        for hyp in hypotheses:
            for i in range(len(hyp) - ngram + 1):
                token = ''.join(hyp[i:(i + ngram)])
                counter[token] += 1
        for value in counter.values():
            ngram_total += value
            ngram_distinct_count += 1
        return (ngram_distinct_count / ngram_total) if ngram_total != 0 else 0.

    return {'DISTINCT {}'.format(v): _calc_distinct(v) for v in range(1, max_ngram + 1)}


class Evaluator:
    def __init__(self, key):
        self.key = key

    def evaluate(self, ref_file, hyp_file, to_lower):
        references = []
        for line in read_json_lines(ref_file):
            ref = line.get(self.key, '').strip().split()  # ref is a list of tokens
            if to_lower:
                ref = list(map(str.lower, ref))
            references.append(ref)

        hypotheses = []
        for line in read_json_lines(hyp_file):
            hyp = line.get(self.key, '').strip().split()  # hyp is a list of tokens
            if to_lower:
                hyp = list(map(str.lower, hyp))
            hypotheses.append(hyp)

        assert len(references) == len(hypotheses)

        results = {}
        results.update(calc_bleu(references, hypotheses))
        results.update(calc_f1(references, hypotheses))
        results.update(calc_distinct_ngram(hypotheses, max_ngram=2))

        for key, value in results.items():
            logger.info('{}: {:>.4f}'.format(key, value))

        return results
