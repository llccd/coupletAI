# coupletAI

参考 <https://github.com/wangjiezju1988/aichpoem> 实现的对对联AI

## Dataset

采用 <https://github.com/wb14123/couplet-dataset>，包含70万首对联

需要手动去除数据集中每个字之间的空格，并合并训练集和测试集。上联保存到`data/in.txt`，下联保存到`data/out.txt`，可以到Release下载处理好的数据集

## BERT Model

采用 <https://github.com/ymcui/Chinese-BERT-wwm> 中的预训练模型`RoBERTa-wwm-ext, Chinese`，模型存放到`data/chinese_roberta_wwm_ext_L-12_H-768_A-12`目录

## Dependencies

- tensorflow 2.X : `pip install tensorflow`
- bert4keras : `pip install bert4keras`

## Training

运行`train_coupletAI.py`来训练模型，如果有多张显卡，可以用多GPU版本`train_coupletAI_multigpu.py`加速训练

每个epoch都会保存最优的权重参数到`data/best_model.weight`（大小约1G），训练结束后运行`save_h5.py`读取最优的权重参数，保存h5文件到`data/model.h5`（大小约350M）

如果不想自己训练，可以到Release下载已经训练好的`model.h5`

## Local deployment

运行`api_server.py`就能启动服务，通过浏览器打开 <http://[::1]:11456> 查看效果
