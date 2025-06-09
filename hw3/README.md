# 第三次作业

大模型下游任务性能评测

## 任务描述

利用北京大学标注的《人民日报》1998年1月份的分词语料，借助外部网站部署的大模型API完成如下任务：

1) 测试并计算一个大模型在随机抽取的50条数据上的平均分词性能；

2) 对于同一批数据，对比分析大模型的参数量大小、提示词(Prompt)等因素对大模型分词性能的影响；

3) 基于上面任务的结果，总结分析提升大模型在分词等下游任务上性能的方法。 

说明：部分大模型API可以从以下网站免费获得

1. https://www.siliconflow.cn

2. https://ai.gitee.com/serverless-api

3. https://bailian.console.aliyun.com/?tab=model#/model-market


## 项目结构

```sh
hw3/
├── README.md                                       # 项目说明文件
├── configs/                                            # 配置文件目录
│   ├── api_config.yaml                                 # api密钥配置(需要自行创建并填写自己的api密钥)
│   ├── exp_config.yaml                                 # 实验配置文件
├── data/                                           # 数据目录    
│   ├── processed_data/     
│   │   ├── sampled_data.json                           # 处理后的数据样本
│   ├── raw_corpus/
│   │   ├── ChineseCorpus199801.txt                     # 原始语料
├── prompts/                                        # 提示词目录 
│   ├── base_zero_shot.txt                              # 基础零-shot提示词 
│   ├── detailed_instruction.txt                        # 详细指令提示词
│   ├── few_shot_example.txt                            # few-shot示例提示词
├── reports/                                        # 报告目录
│   ├── report.md
├── requirements.txt                                # 依赖包列表 
├── results/                                        # 结果目录
│   ├── visualizations/                                 # 可视化结果目录
├── src/                                            # 源代码目录
│   ├── data_process.py                                 # 数据处理脚本
│   ├── evaluator.py                                    # 评估器脚本
│   ├── llm_caller.py                                   # LLM调用脚本
│   ├── main.py                                         # 主程序脚本    
│   ├── visualizer.py                                   # 可视化脚本
```


## 使用方法：

1. 安装依赖：`pip install -r requirements.txt`
2. 配置API密钥：编辑`configs/api_config.yaml`文件，填写自己的API密钥。
3. 运行主程序：`python src/main.py`

## 实验配置说明

见实验配置文件`configs/exp_config.yaml`中的注释说明。

### api密钥配置说明

`configs/api_config.yaml`文件中需要填写自己的API密钥，格式如下：

```yaml
api_key : 'sk-put-your-api-key-here'
```

