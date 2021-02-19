#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_KERAS'] = '1'

from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from bert4keras.snippets import DataGenerator
from keras.models import Model
from shared import CrossEntropy, config_path, checkpoint_path, dict_path
from shared.decoder import BertDecoder

# 基本参数
maxlen = 67
batch_size = 16
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
    """
    def __iter__(self, _):
        batch_token_ids, batch_segment_ids = [], []
        fpI = open(self.data + '/in.txt', 'r' , encoding='utf-8')
        fpO = open(self.data + '/out.txt', 'r' , encoding='utf-8')
        for lineI in fpI:
            lineI = lineI.rstrip()
            lineO = fpO.readline().rstrip()
            token_ids, segment_ids = tokenizer.encode(
                lineI, lineO, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            token_ids, segment_ids = tokenizer.encode(
                lineO, lineI, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []
        if batch_token_ids:
            batch_token_ids = sequence_padding(batch_token_ids)
            batch_segment_ids = sequence_padding(batch_segment_ids)
            yield [batch_token_ids, batch_segment_ids], None

model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()

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

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

