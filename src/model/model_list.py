# -*- coding: utf-8 -*-

"""
@Author             : Bao
@Date               : 2020/2/20 20:42
@Desc               :
@Last modified by   : Bao
@Last modified date : 2020/6/28 13:57
"""

from .seq2seq import Seq2Seq
from .ptrnet_h import PtrNetH
from .ptrnet_k import PtrNetK
from .dual_ptrnet import DualPtrNet

model_list = {
    'seq2seq': Seq2Seq,
    'ptrnet_h': PtrNetH,
    'ptrnet_k': PtrNetK,
    'dual_ptrnet': DualPtrNet,
}


def get_model(config, embedding_matrix=None):
    assert config.current_model in model_list

    return model_list[config.current_model](config, embedding_matrix)
