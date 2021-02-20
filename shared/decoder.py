# -*- coding: utf-8 -*-

import numpy as np

class BertDecoder:
    def generate(self, s, topk=2, rand=False):
        """beam search解码
        每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
        """
        token_ids, segment_ids = self.tokenizer.encode(s)
        target_ids = [[] for _ in range(topk)]  # 候选答案id
        target_scores = [0] * topk  # 候选答案分数

        for i in range(len(token_ids) - 2):  # 强制要求输出不超过输入长度
            _target_ids = np.array([token_ids + t for t in target_ids])
            _segment_ids = np.array([segment_ids + [1] * len(t) for t in target_ids])
            _probas = self.model.predict([_target_ids, _segment_ids])[:, -1, 4:]  # 直接忽略[PAD], [UNK], [CLS]
            _log_probas = np.log(_probas + 1e-6)  # 取对数，方便计算
            _arg = _log_probas.argsort(axis=1)[:, -64:]

            for j in range(topk):
                if token_ids[i+1] in token_ids[1:i+1]:
                    # 第i个值的输入之前出现过，奖励
                    idxs = [ii for ii,x in enumerate(token_ids[1:i+1]) if x==token_ids[i+1]]
                    target_idxs = [target_ids[j][ii] for ii in idxs]
                else:
                    idxs = []
                    target_idxs = []

                for k in _arg[j]:
                    if k + 4 not in self.comma and k + 4 in token_ids:
                        # 当前不为标点，如果在输入里面引入惩罚
                        _log_probas[j][k] = (_log_probas[j][k]-1) * 1000
                    elif k + 4 in self.comma and token_ids[i+1] not in self.comma:
                        # 当前为标点，但是输入不是，引入惩罚
                        _log_probas[j][k] = (_log_probas[j][k]-1) * 1000
                    elif k + 4 in self.comma and token_ids[i+1] in self.comma and k+4 == token_ids[i+1]:
                        # 当前为标点，预测也为标点，奖励
                        _log_probas[j][k] = _log_probas[j][k] / 10
                    elif k + 4 in target_idxs:
                        _log_probas[j][k] = _log_probas[j][k] / 1000
                    elif k + 4 in target_ids[j]:
                        # 如果第i个值的输入之前没有出现过，惩罚
                        _log_probas[j][k] = (_log_probas[j][k]-1) * 1000

            _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]  # 每一项选出topk
            _candidate_ids, _candidate_scores = [], []
            for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
                    # 预测第一个字的时候，输入的topk事实上都是同一个，
                    # 所以只需要看第一个，不需要遍历后面的。
                    if i == 0 and j > 0:
                            break
                    for k in _topk_arg[j]:
                            _candidate_ids.append(ids + [k + 4])
                            _candidate_scores.append(sco + _log_probas[j][k])

            _arg = np.argsort(_candidate_scores)
            _topk_arg = np.argsort(_candidate_scores)[-topk:]  # 从中选出新的topk
            target_ids = [_candidate_ids[k] for k in _topk_arg]
            target_scores = [_candidate_scores[k] for k in _topk_arg]
        return self.tokenizer.decode(target_ids[np.random.randint(topk) if rand else -1])

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.comma, _ = self.tokenizer.encode('，。？！、；：')
        self.comma = self.comma[1:-1]

