#! -*- coding: utf-8 -*-
# SimBERT base 基本例子
# 测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.7.7
import os
os.environ['TF_KERAS'] = '1'
import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
# from bert4keras.snippets import uniout
from keras.layers import *

maxlen = 32

# bert配置
config_path = 'chinese_simbert_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'chinese_simbert_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'chinese_simbert_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器


bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
    hierarchical_position=True  # 文本长度大于512时
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


def simbert_encoder(text):
    if isinstance(text, list):
        r = text
    elif isinstance(text, str):
        r = [text]
    else:
        r = []

    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z ** 2).sum(axis=1, keepdims=True) ** 0.5  # 归一化
    return Z


if __name__ == '__main__':
    text = "晚上经常出汗，多梦，这是什么问题"
    vector = simbert_encoder(text)
    print(vector.tolist())
    print(type(vector))
    print(vector.shape)