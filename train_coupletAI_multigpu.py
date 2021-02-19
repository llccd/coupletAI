#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_KERAS'] = '1' 

from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import DataGenerator
from keras.models import Model
import tensorflow as tf
from shared import CrossEntropy, config_path, checkpoint_path, dict_path
from shared.decoder import BertDecoder

# 基本参数
maxlen = 67
batch_size = 64
steps_per_epoch = 1000
epochs = 10000

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    （每次只需要返回一条样本）
    """
    def __iter__(self, _):
        fpI = open(self.data + '/in.txt', 'r' , encoding='utf-8')
        fpO = open(self.data + '/out.txt', 'r' , encoding='utf-8')
        for lineI in fpI:
            lineI = lineI.rstrip()
            lineO = fpO.readline().rstrip()
            token_ids, segment_ids = tokenizer.encode(
                lineI, lineO, maxlen=maxlen
            )
            yield token_ids, segment_ids
            token_ids, segment_ids = tokenizer.encode(
                lineO, lineI, maxlen=maxlen
            )
            yield token_ids, segment_ids

strategy = tf.distribute.MirroredStrategy()  # 建立单机多卡策略

with strategy.scope():  # 调用该策略
    bert = build_transformer_model(
        config_path,
        checkpoint_path=None,  # 此时可以不加载预训练权重
        application='unilm',
        keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
        return_keras_model=False,  # 返回bert4keras类，而不是keras模型
    )

    model = bert.model  # 这个才是keras模型
    output = CrossEntropy(2)(model.inputs + model.outputs)

    model = Model(model.inputs, output)
    model.compile(optimizer=Adam(1e-5))
    model.summary()
    bert.load_weights_from_checkpoint(checkpoint_path)  # 必须最后才加载预训练权重

decoder = BertDecoder(tokenizer, model)

class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('data/best_model.weights')
        # 演示效果
        s1 = u'要独裁，残杀学生之政府，从来没有好结果'
        s2 = u'半月朦胧全斗焕'
        print()
        for s in [s1, s2]:
            print(u'对联:', decoder.generate(s))
        print()

if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = data_generator('data', batch_size)
    dataset = train_generator.to_dataset(
        types=('float32', 'float32'),
        shapes=([None], [None]),  # 配合后面的padded_batch=True，实现自动padding
        names=('Input-Token', 'Input-Segment'),
        padded_batch=True
    )  # 数据要转为tf.data.Dataset格式，names跟输入层的名字对应

    model.fit(
        dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

