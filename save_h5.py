#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_KERAS'] = '1' 

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import load_vocab
from shared import CrossEntropy, config_path, checkpoint_path, dict_path
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

_, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = tf.keras.models.Model(model.inputs, output)
model.load_weights('data/best_model.weights')
model.save('data/model.h5')
