import json
import os
import time
import yaml
from glob import glob
from datetime import datetime
from typing import List, Dict

from data_process import load_and_sample_corpus, save_sampled_data
from llm_caller import get_llm_segmentation
from evaluator import calculate_metrics
from visualizer import (create_comparison_charts, save_results_summary, 
                       print_performance_analysis)

# --- 配置路径 ---
CONFIG_DIR = r"..\configs"
EXP_CONFIG_PATH = os.path.join(CONFIG_DIR, "exp_config.yaml")
API_CONFIG_PATH = os.path.join(CONFIG_DIR, "api_config.yaml")
PROMPT_DIR = r"..\prompts"

# --- 配置加载函数 ---
def load_yaml_config(config_path):
    """从YAML文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"加载配置文件 {config_path} 失败: {e}")
        return {}

def load_prompt_templates():
    """从prompts目录加载所有提示词模板"""
    templates = {}
    prompt_files = glob(os.path.join(PROMPT_DIR, "*.txt"))
    
    if not prompt_files:
        print(f"警告: 未在 {PROMPT_DIR} 目录下找到提示词文件，使用默认提示词")
        # 提供默认的提示词
        templates["base_zero_shot"] = "请对以下句子进行中文分词，分词结果用空格隔开：\n句子：{sentence}\n分词结果："
        return templates
        
    for file_path in prompt_files:
        try:
            template_name = os.path.splitext(os.path.basename(file_path))[0]
            with open(file_path, 'r', encoding='utf-8') as f:
                template_content = f.read().strip()
                templates[template_name] = template_content
            print(f"已加载提示词模板: {template_name}")
        except Exception as e:
            print(f"加载提示词模板 {file_path} 失败: {e}")
    
    return templates

# --- 加载配置 ---
# 加载实验配置
exp_config = load_yaml_config(EXP_CONFIG_PATH)
# 加载API配置
api_config = load_yaml_config(API_CONFIG_PATH)
# 加载提示词模板
PROMPT_TEMPLATES = load_prompt_templates()

# 从配置文件中读取数据路径和实验参数
RAW_CORPUS_PATH = exp_config.get('corpus_file_path', r"..\data\raw_corpus\ChineseCorpus199801.txt")
SAMPLED_DATA_PATH = exp_config.get('sampled_output_path', r"..\data\processed_data\sampled_data.json")
NUM_SAMPLES = exp_config.get('num_samples', 50)
NUM_SAMPLES_COMPARISON = exp_config.get('num_samples_comparison', 10)
API_CALL_DELAY_SECONDS = exp_config.get('api_call_delay_seconds', 2)

# 设置结果输出目录
RESULTS_DIR = r"..\results"
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, "visualizations")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# 从配置文件读取模型配置
MODEL_CONFIGS = exp_config.get('model_configs', {})

# --- 辅助函数 ---
def run_single_experiment(experiment_id: str, data: list[dict], model_name: str, 
                         prompt_template: str, model_id: str = None, 
                         prompt_type: str = None, param_size: str = None):
    """
    对给定的数据集，使用指定的模型和提示词进行分词并评估。
    """
    print(f"\n--- 开始实验: {experiment_id} ---")
    print(f"模型: {model_name}")
    print(f"提示词类型: {prompt_type}")
    if param_size:
        print(f"参数量: {param_size}")

    results = []
    all_metrics = {"precision": [], "recall": [], "f1_score": []}

    for i, item in enumerate(data):
        sentence = item["original_sentence"]
        standard_seg = item["standard_segmentation"]
        
        print(f"  处理样本 {i+1}/{len(data)}: '{sentence[:30]}...'")
        
        model_seg = get_llm_segmentation(prompt_template, sentence, model_name)
        time.sleep(API_CALL_DELAY_SECONDS) # API调用间隔

        if model_seg is None:
            print(f"    警告: 未能获取模型分词结果。跳过此样本。")
            metrics = {"precision": 0, "recall": 0, "f1_score": 0, "correct_segments": 0, 
                      "standard_total": len(standard_seg), "model_total": 0, "error": "API_CALL_FAILED"}
        else:
            metrics = calculate_metrics(standard_seg, model_seg)
        
        results.append({
            "original_sentence": sentence,
            "standard_segmentation": standard_seg,
            "model_segmentation": model_seg if model_seg else [],
            "metrics": metrics
        })
        
        if model_seg is not None: # 只对成功获取结果的样本计算平均指标
            all_metrics["precision"].append(metrics["precision"])
            all_metrics["recall"].append(metrics["recall"])
            all_metrics["f1_score"].append(metrics["f1_score"])

    # 计算平均指标
    avg_metrics = {
        "avg_precision": sum(all_metrics["precision"]) / len(all_metrics["precision"]) if all_metrics["precision"] else 0,
        "avg_recall": sum(all_metrics["recall"]) / len(all_metrics["recall"]) if all_metrics["recall"] else 0,
        "avg_f1_score": sum(all_metrics["f1_score"]) / len(all_metrics["f1_score"]) if all_metrics["f1_score"] else 0,
        "num_successful_samples": len(all_metrics["precision"]),
        "num_total_samples": len(data)
    }
    
    print(f"实验 {experiment_id} 完成。平均指标: P={avg_metrics['avg_precision']:.4f}, R={avg_metrics['avg_recall']:.4f}, F1={avg_metrics['avg_f1_score']:.4f}")

    # 保存结果
    output_filename = os.path.join(RESULTS_DIR, f"{experiment_id}_results.json")
    
    experiment_result = {
        "experiment_details": {
            "id": experiment_id, 
            "model": model_name, 
            "prompt_snippet": prompt_template[:100],
            "model_id": model_id,
            "prompt_type": prompt_type,
            "param_size": param_size
        },
        "average_metrics": avg_metrics, 
        "individual_results": results
    }
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(experiment_result, f, ensure_ascii=False, indent=4)
        print(f"详细结果已保存到: {output_filename}")
    except IOError:
        print(f"错误: 无法写入结果文件 {output_filename}")
    
    # 返回包含实验信息的结果
    return {
        "experiment_id": experiment_id,
        "model_name": model_name,
        "model_id": model_id,
        "prompt_type": prompt_type,
        "param_size": param_size,
        "avg_metrics": avg_metrics,
        "detailed_results": results
    }

def run_model_comparison_experiments(data: List[Dict]) -> List[Dict]:
    """运行模型参数量比较实验"""
    print("\n========== 开始任务2A: 模型参数量比较实验 ==========")
    
    model_comparison_results = []
    model_configs = MODEL_CONFIGS.get("task2_models_param_comparison", [])
    
    if not model_configs:
        print("警告: 未配置模型比较实验")
        return []
    
    for model_config in model_configs:
        model_id = model_config.get("id", "unknown")
        model_name = model_config.get("model_name", "")
        param_size = model_config.get("param_size_desc", "unknown")
        prompt_key = model_config.get("prompt_key", "detailed_instruction")
        
        prompt_template = PROMPT_TEMPLATES.get(prompt_key, "")
        if not prompt_template:
            print(f"警告: 未找到提示词模板 {prompt_key}，跳过模型 {model_id}")
            continue
        
        experiment_id = f"model_comparison_{model_id}"
        
        result = run_single_experiment(
            experiment_id=experiment_id,
            data=data,
            model_name=model_name,
            prompt_template=prompt_template,
            model_id=model_id,
            prompt_type=prompt_key,
            param_size=param_size
        )
        
        model_comparison_results.append(result)
    
    return model_comparison_results

def run_prompt_comparison_experiments(data: List[Dict]) -> List[Dict]:
    """运行提示词比较实验"""
    print("\n========== 开始任务2B: 提示词策略比较实验 ==========")
    
    prompt_comparison_results = []
    prompt_model_config = MODEL_CONFIGS.get("task2_prompts_comparison_model", {})
    prompts_to_compare = exp_config.get("prompts_to_compare", [])
    
    if not prompt_model_config or not prompts_to_compare:
        print("警告: 未配置提示词比较实验")
        return []
    
    model_name = prompt_model_config.get("model_name", "")
    param_size = prompt_model_config.get("param_size_desc", "unknown")
    
    for prompt_key in prompts_to_compare:
        prompt_template = PROMPT_TEMPLATES.get(prompt_key, "")
        if not prompt_template:
            print(f"警告: 未找到提示词模板 {prompt_key}，跳过此提示词")
            continue
        
        experiment_id = f"prompt_comparison_{prompt_key}"
        
        result = run_single_experiment(
            experiment_id=experiment_id,
            data=data,
            model_name=model_name,
            prompt_template=prompt_template,
            model_id="prompt_comparison_model",
            prompt_type=prompt_key,
            param_size=param_size
        )
        
        prompt_comparison_results.append(result)
    
    return prompt_comparison_results

def generate_visualizations(model_results: List[Dict], prompt_results: List[Dict], 
                          all_results: List[Dict]):
    """生成可视化图表"""
    print("\n========== 生成可视化图表 ==========")
    
    visualization_data = {
        "all_experiments": all_results
    }
    
    if model_results:
        visualization_data["model_comparison"] = model_results
    
    if prompt_results:
        visualization_data["prompt_comparison"] = prompt_results
    
    try:
        # 生成图表
        create_comparison_charts(visualization_data, VISUALIZATION_DIR)
        
        # 保存CSV摘要
        save_results_summary(all_results, VISUALIZATION_DIR)
        
        # 打印分析报告
        print_performance_analysis(all_results)
        
        print(f"可视化图表已保存到: {VISUALIZATION_DIR}")
        
    except Exception as e:
        print(f"生成可视化图表时出错: {e}")

# --- 主逻辑 ---
def main():
    print("=" * 60)
    print("             大模型下游任务性能评测实验")
    print("=" * 60)
    
    # 1. 准备数据
    if not os.path.exists(SAMPLED_DATA_PATH):
        print(f"{SAMPLED_DATA_PATH} 不存在，正在从原始语料创建...")
        sampled_data = load_and_sample_corpus(RAW_CORPUS_PATH, NUM_SAMPLES)
        if not sampled_data:
            print("错误: 未能加载或抽样数据。请检查 data_processor.py 和语料路径。")
            return
        save_sampled_data(sampled_data, SAMPLED_DATA_PATH)
    else:
        print(f"从 {SAMPLED_DATA_PATH} 加载已抽样的数据...")
        try:
            with open(SAMPLED_DATA_PATH, 'r', encoding='utf-8') as f:
                sampled_data = json.load(f)
            if len(sampled_data) != NUM_SAMPLES:
                print(f"警告: 文件中的样本数 ({len(sampled_data)}) 与期望的样本数 ({NUM_SAMPLES}) 不符。")
                print(f"{SAMPLED_DATA_PATH} 不符合要求，正在从原始语料创建...")
                sampled_data = load_and_sample_corpus(RAW_CORPUS_PATH, NUM_SAMPLES)
                if not sampled_data:
                    print("错误: 未能加载或抽样数据。请检查 data_processor.py 和语料路径。")
                    return
                save_sampled_data(sampled_data, SAMPLED_DATA_PATH)
        except Exception as e:
            print(f"加载抽样数据失败: {e}。请尝试重新生成或检查文件。")
            return
    
    if not sampled_data:
        print("未能获取测试数据，实验终止。")
        return
    
    print(f"成功加载 {len(sampled_data)} 条测试数据。")

    # 存储所有实验结果
    all_experiment_results = []

    # --- 任务1: 测试一个大模型在随机抽取的50条数据上的平均分词性能 ---
    print("\n========== 开始任务1: 基础模型性能测试 ==========")
    task1_config = MODEL_CONFIGS.get("task1_base_model", {})
    task1_prompt_key = task1_config.get("prompt_key", "detailed_instruction")
    task1_prompt = PROMPT_TEMPLATES.get(task1_prompt_key, "")
    
    if task1_prompt and task1_config.get("model_name"):
        task1_result = run_single_experiment(
            experiment_id="task1_base_performance",
            data=sampled_data,
            model_name=task1_config.get("model_name", ""),
            prompt_template=task1_prompt,
            model_id="task1_base_model",
            prompt_type=task1_prompt_key,
            param_size=task1_config.get("param_size_desc", "unknown")
        )
        all_experiment_results.append(task1_result)
    else:
        print("警告: 任务1配置不完整，跳过基础性能测试")

    # 对于任务2，缩减样本量以加快实验速度
    sampled_data = sampled_data[:NUM_SAMPLES_COMPARISON] if len(sampled_data) > NUM_SAMPLES_COMPARISON else sampled_data

    # --- 任务2A: 不同参数量模型比较 ---
    print("\n========== 开始任务2A: 模型参数量比较实验 ==========")
    model_comparison_results = run_model_comparison_experiments(sampled_data)
    all_experiment_results.extend(model_comparison_results)

    # --- 任务2B: 不同提示词策略比较 ---
    print("\n========== 开始任务2B: 提示词策略比较实验 ==========")
    prompt_comparison_results = run_prompt_comparison_experiments(sampled_data)
    all_experiment_results.extend(prompt_comparison_results)

    # --- 任务3: 数据可视化和分析 ---
    if all_experiment_results:
        generate_visualizations(model_comparison_results, prompt_comparison_results, 
                              all_experiment_results)
    else:
        print("警告: 没有实验结果，跳过可视化生成")

    print("\n========== 所有实验完成 ==========")
    print(f"实验结果保存在目录: {RESULTS_DIR}")
    print(f"可视化图表保存在目录: {VISUALIZATION_DIR}")
    print("\n实验总结:")
    print(f"- 总共完成 {len(all_experiment_results)} 个实验")
    print(f"- 模型比较实验: {len(model_comparison_results)} 个")
    print(f"- 提示词比较实验: {len(prompt_comparison_results)} 个")

if __name__ == '__main__':
    # 检查SiliconFlow API密钥是否已配置
    siliconflow_api_key = api_config.get('api_key', '')
    
    if not siliconflow_api_key or siliconflow_api_key == "your_api_key_here":
        print(f"警告: SiliconFlow API密钥未在配置文件中设置或使用了默认占位符。")
        print("请更新 configs/api_config.yaml 文件中的API密钥。")
        user_choice = input("是否继续运行 (API调用可能会失败)? (y/n): ")
        if user_choice.lower() != 'y':
            print("实验终止。")
            exit()
            
    main()
