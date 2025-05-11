# 第二次作业

词向量的训练与分析

## 1. 任务描述

请下载调试FNN、RNN和LSTM模型的开源工具。利用北京大学标注的《人民日报》1998年1月份的分词语料，或者利用网络爬虫自己从互联网上收集足够多的英文文本语料，借助 FNN 或者 RNN/LSTM 开源工具，完成如下任务：

1) 获得汉语或英语词语的词向量。
2) 在任务(1)的基础上，随机选择20个单词，计算与其词向量最相似的前10个单词。
3) 对于同一批词汇，对比分别用 FNN，RNN 或 LSTM 获得的词向量的差异。


说明：

1.  如果计算资源的限制，神经网络参数不必选择过大，例如：词表选择1000个左右单词即可，其余单词用<UNK>代替；词向量的维度可设为10左右；神经网络的层数设置为1到2层；
2.  可以使用某一种开放的深度学习框架，如TensorFlow或者PyTorch。
3.  如果不借助开源工具和开放的深度学习框架，题目中的任务(3)可以不做。

## 项目结构

```
hw2/
├── README.md                           # 说明文档
├── config.yaml                         # 配置文件 
├── environment.yaml                    # 环境信息 
├── data/                               # 数据文件夹      
│   ├── ChineseCorpus199801-GB2312.txt      # 原始数据
│   ├── ChineseCorpus199801.txt             # 转换为utf-8编码后的数据
│   ├── word_frequency.txt                  # 词频统计文件  
├── main.py                                 # 主程序 
├── models/                                 # 模型文件夹
│   ├── FNN.py                              # 前馈神经网络模型  
│   ├── LSTM.py                             # 长短时记忆网络模型
│   ├── RNN.py                              # 循环神经网络模型
│   ├── __init__.py
├── results/                            # 结果文件夹
│   ├── word_vectors/                       # 词向量文件夹
|       ├── *_vectors.txt                       # 词向量文件
│       ├── *_vectors.npz                       # 词向量文件(numpy数组)
│   ├── *_model.pt                      # 模型参数
│   ├── *_word_vectors.png              # 模型词向量可视化(对应问题1)
│   ├── similar_words.txt                   # 相似词汇(对应问题2)
│   ├── vector_comparison.txt               # 不同模型词向量对比(对应问题3)
│   ├── loss_history.png                    # 训练损失曲线图
│   ├── loss_history.csv                    # 训练损失曲线数据
│   ├── 
├── utils/                              # 工具文件夹
│   ├── __init__.py
│   ├── load.py                             # 数据加载工具
│   ├── process.py                          # 数据预处理工具
├── reports/                            # 报告文件夹
│   ├── report.pdf                          # 实验报告
│   ├── report.md                           # 实验报告markdown文件
```



## 实验环境

python版本：3.12.9
pytorch版本：2.7.0+cu128

在windows 11下测试通过，Linux下未测试。

为方便使用，提供了`environment.yaml`文件，可以使用conda创建虚拟环境：

```bash
conda env create -f environment.yaml -n <env_name>
```

然后激活环境：

```bash
conda activate <env_name>
```

之后就可以运行`main.py`文件了。

```bash
python main.py
```