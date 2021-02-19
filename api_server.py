#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_KERAS'] = '1'

import time
from flask import Flask, request, jsonify, make_response
from bert4keras.backend import gelu_erf
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.layers import PositionEmbedding, LayerNormalization, MultiHeadAttention, FeedForward, BiasAdd, Loss
import tensorflow as tf
from shared import CrossEntropy, dict_path
from shared.decoder import BertDecoder

tf.config.set_visible_devices([], 'GPU') # 使用CPU
#os.environ['CUDA_VISIBLE_DEVICES']='0' # 指定GPU 0
#physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

app = Flask(__name__)

token_dict, _ = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)
model = tf.keras.models.load_model('data/model.h5', custom_objects={'PositionEmbedding': PositionEmbedding, 'gelu_erf':gelu_erf, 'LayerNormalization':LayerNormalization, 'MultiHeadAttention':MultiHeadAttention, 'FeedForward':FeedForward, 'BiasAdd':BiasAdd, 'CrossEntropy':CrossEntropy}, compile=False)

decoder = BertDecoder(tokenizer, model)

# seq2seq请求
@app.route("/api/v1/couplet", methods=['GET','POST','OPTIONS'])
def bertseq2seq():
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_prelight_response()
    start =time.time()
    data = request.values['data'] if 'data' in request.values else '虎啸青山抒壮志'
    topk = int(request.values['topk']) if 'topk' in request.values else 1
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'topk:', topk)
    r = decoder.generate(data,topk)
    end = time.time()
    print('In：%s' %(data))
    print('Out：%s' %(r))
    res={'result':r, 'timeused':int(1000 * (end - start))}
    print('使用时间：%sms\n---------' %(res['timeused']))
    return _corsify_actual_response(jsonify(res))

# 跨域支持
def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

def _build_cors_prelight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def start_web_server(host='::', port=11456):
    app.run(host, port)

if __name__ == "__main__":
    start_web_server()

