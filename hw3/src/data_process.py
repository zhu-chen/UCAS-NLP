import random
import json
import re
import yaml 
import os 
import sys 

def parse_line(line: str) -> tuple[str, list[str]]:
    words_with_pos = re.split(r'\s+', line.strip())
    segmented_words = []
    original_sentence_parts = []
    for item in words_with_pos:
        if not item:
            continue
        word = item.split('/')[0]
        segmented_words.append(word)
        original_sentence_parts.append(word)
    original_sentence = "".join(original_sentence_parts) 
    return original_sentence, segmented_words

def load_and_sample_corpus(corpus_path: str, sample_size: int) -> list[dict]:
    
    sampled_data = []
    all_lines = []
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f: 
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 语料文件 {corpus_path} 未找到。")
        return []
    except Exception as e:
        print(f"读取语料文件时发生错误: {e}")
        return []

    if not all_lines:
        print("错误: 语料文件为空或读取失败。")
        return []

    # 确保抽样数量不超过总行数
    actual_sample_size = min(sample_size, len(all_lines))
    if actual_sample_size < sample_size:
        print(f"警告: 请求抽样 {sample_size} 条，但语料中只有 {len(all_lines)} 条。将抽样 {actual_sample_size} 条。")

    sampled_lines = random.sample(all_lines, actual_sample_size)

    for line in sampled_lines:
        line = line.strip()
        if not line:
            continue
        try:
            original_sentence, standard_segmentation = parse_line(line)
            if original_sentence and standard_segmentation:
                sampled_data.append({
                    "original_sentence": original_sentence,
                    "standard_segmentation": standard_segmentation
                })
        except Exception as e:
            print(f"解析行时发生错误: '{line[:50]}...' - {e}")
            continue # 跳过无法解析的行

    return sampled_data

def save_sampled_data(data: list[dict], output_path: str):
    """
    将抽样的数据保存到JSON文件。
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"抽样数据已保存到: {output_path}")
    except IOError:
        print(f"错误: 无法写入文件 {output_path}")

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"配置已从 {config_path} 加载。")
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件 {config_path} 未找到。")
        return None
    except yaml.YAMLError as e:
        print(f"错误: 解析配置文件 {config_path} 失败: {e}")
        return None

if __name__ == '__main__':
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    config_file_path = r"../configs/exp_config.yaml"
    config = load_config(config_file_path)

    if not config:
        print("无法加载配置，程序退出。")
        sys.exit(1) # 退出程序


    corpus_file_path = config.get('corpus_file_path')
    sampled_output_path = config.get('sampled_output_path')
    num_samples = config.get('num_samples', 50)
    
    print(f"正在从 {corpus_file_path} 加载和抽样数据...")
    sampled_items = load_and_sample_corpus(corpus_file_path, num_samples)
    


    if sampled_items:
        save_sampled_data(sampled_items, sampled_output_path)
        print(f"成功抽取 {len(sampled_items)} 条数据。")
        # 打印一些样本以供检查
        for i, item in enumerate(sampled_items[:3]):
            print(f"\n样本 {i+1}:")
            print(f"  原始句子: {item['original_sentence']}")
            print(f"  标准分词: {item['standard_segmentation']}")
    else:
        print("未能抽取数据。")

