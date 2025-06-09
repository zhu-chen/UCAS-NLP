import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_comparison_charts(results_data: Dict, output_dir: str):
    """
    创建模型和prompt比较的可视化图表
    
    Args:
        results_data: 包含所有实验结果的字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 模型参数量vs性能比较
    if 'model_comparison' in results_data:
        create_model_comparison_chart(results_data['model_comparison'], output_dir)
    
    # 2. Prompt策略vs性能比较 
    if 'prompt_comparison' in results_data:
        create_prompt_comparison_chart(results_data['prompt_comparison'], output_dir)
    
    # 3. 综合性能对比雷达图
    if 'all_experiments' in results_data:
        create_comprehensive_radar_chart(results_data['all_experiments'], output_dir)

def create_model_comparison_chart(model_results: List[Dict], output_dir: str):
    """创建模型参数量vs性能比较图"""
    
    # 提取数据
    model_names = []
    param_sizes = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for result in model_results:
        model_names.append(result['model_id'])
        param_sizes.append(float(result['param_size'].replace('B', '')))
        f1_scores.append(result['avg_metrics']['avg_f1_score'])
        precisions.append(result['avg_metrics']['avg_precision'])
        recalls.append(result['avg_metrics']['avg_recall'])
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 参数量vs F1分数散点图
    ax1.scatter(param_sizes, f1_scores, s=100, alpha=0.7, c='blue')
    for i, name in enumerate(model_names):
        ax1.annotate(name.split('_')[-1], (param_sizes[i], f1_scores[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax1.set_xlabel('模型参数量 (B)')
    ax1.set_ylabel('F1分数')
    ax1.set_title('模型参数量与F1性能关系')
    ax1.grid(True, alpha=0.3)
    
    # 2. 三个指标的柱状图比较
    x = np.arange(len(model_names))
    width = 0.25
    
    ax2.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    ax2.bar(x, recalls, width, label='Recall', alpha=0.8)
    ax2.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax2.set_xlabel('模型')
    ax2.set_ylabel('性能指标')
    ax2.set_title('不同模型的性能指标对比')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name.split('_')[-1] for name in model_names])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 性能提升趋势线
    sorted_indices = np.argsort(param_sizes)
    sorted_params = [param_sizes[i] for i in sorted_indices]
    sorted_f1 = [f1_scores[i] for i in sorted_indices]
    
    ax3.plot(sorted_params, sorted_f1, 'o-', linewidth=2, markersize=8)
    ax3.set_xlabel('模型参数量 (B)')
    ax3.set_ylabel('F1分数')
    ax3.set_title('参数量增加与性能提升趋势')
    ax3.grid(True, alpha=0.3)
    
    # 4. 模型效率分析（性能/参数量）
    efficiency = [f1/param for f1, param in zip(f1_scores, param_sizes)]
    ax4.bar(range(len(model_names)), efficiency, alpha=0.7, color='green')
    ax4.set_xlabel('模型')
    ax4.set_ylabel('效率 (F1分数/参数量B)')
    ax4.set_title('模型效率对比')
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels([name.split('_')[-1] for name in model_names], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_prompt_comparison_chart(prompt_results: List[Dict], output_dir: str):
    """创建prompt策略vs性能比较图"""
    
    # 提取数据
    prompt_names = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for result in prompt_results:
        prompt_names.append(result['prompt_type'])
        f1_scores.append(result['avg_metrics']['avg_f1_score'])
        precisions.append(result['avg_metrics']['avg_precision'])
        recalls.append(result['avg_metrics']['avg_recall'])
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 三个指标的柱状图
    x = np.arange(len(prompt_names))
    width = 0.25
    
    ax1.bar(x - width, precisions, width, label='Precision', alpha=0.8, color='skyblue')
    ax1.bar(x, recalls, width, label='Recall', alpha=0.8, color='lightgreen')
    ax1.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='salmon')
    
    ax1.set_xlabel('Prompt策略')
    ax1.set_ylabel('性能指标')
    ax1.set_title('不同Prompt策略的性能对比')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace('_', '\n') for name in prompt_names])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. F1分数排序
    sorted_indices = np.argsort(f1_scores)[::-1]
    sorted_prompts = [prompt_names[i] for i in sorted_indices]
    sorted_f1 = [f1_scores[i] for i in sorted_indices]
    
    colors = ['gold', 'silver', '#CD7F32']  # 金银铜色
    ax2.bar(range(len(sorted_prompts)), sorted_f1, 
           color=colors[:len(sorted_prompts)], alpha=0.8)
    ax2.set_xlabel('Prompt策略 (按F1分数排序)')
    ax2.set_ylabel('F1分数')
    ax2.set_title('Prompt策略性能排名')
    ax2.set_xticks(range(len(sorted_prompts)))
    ax2.set_xticklabels([name.replace('_', '\n') for name in sorted_prompts])
    
    # 添加数值标签
    for i, v in enumerate(sorted_f1):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    ax2.grid(True, alpha=0.3)
    
    # 3. 雷达图
    categories = ['Precision', 'Recall', 'F1-Score']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    
    for i, (prompt, prec, rec, f1) in enumerate(zip(prompt_names, precisions, recalls, f1_scores)):
        values = [prec, rec, f1]
        values += values[:1]  # 闭合图形
        ax3.plot(angles, values, 'o-', linewidth=2, label=prompt.replace('_', ' '))
        ax3.fill(angles, values, alpha=0.25)
    
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 1)
    ax3.set_title('Prompt策略综合性能雷达图')
    ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 4. 改进潜力分析
    improvements = []
    baselines = min(f1_scores)
    for f1 in f1_scores:
        improvement = (f1 - baselines) / baselines * 100 if baselines > 0 else 0
        improvements.append(improvement)
    
    ax4.bar(range(len(prompt_names)), improvements, alpha=0.7, color='purple')
    ax4.set_xlabel('Prompt策略')
    ax4.set_ylabel('相对于最低性能的提升 (%)')
    ax4.set_title('Prompt策略改进效果')
    ax4.set_xticks(range(len(prompt_names)))
    ax4.set_xticklabels([name.replace('_', '\n') for name in prompt_names])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prompt_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_radar_chart(all_results: List[Dict], output_dir: str):
    """创建综合性能对比雷达图"""
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    categories = ['Precision', 'Recall', 'F1-Score']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_results)))
    
    for i, result in enumerate(all_results):
        values = [
            result['avg_metrics']['avg_precision'],
            result['avg_metrics']['avg_recall'], 
            result['avg_metrics']['avg_f1_score']
        ]
        values += values[:1]
        
        label = f"{result.get('model_id', result.get('prompt_type', f'实验{i+1}'))}"
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('所有实验综合性能对比', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_results_summary(all_results: List[Dict], output_dir: str):
    """保存结果摘要为CSV文件"""
    
    summary_data = []
    for result in all_results:
        summary_data.append({
            '实验ID': result.get('experiment_id', '未知'),
            '模型': result.get('model_name', '未知'),
            'Prompt类型': result.get('prompt_type', '未知'),
            '参数量': result.get('param_size', '未知'),
            'Precision': result['avg_metrics']['avg_precision'],
            'Recall': result['avg_metrics']['avg_recall'],
            'F1-Score': result['avg_metrics']['avg_f1_score'],
            '成功样本数': result['avg_metrics']['num_successful_samples'],
            '总样本数': result['avg_metrics']['num_total_samples']
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(output_dir, 'results_summary.csv'), 
              index=False, encoding='utf-8-sig')
    
    return df

def print_performance_analysis(all_results: List[Dict]):
    """打印性能分析报告"""
    
    print("\n" + "="*60)
    print("                     性能分析报告")
    print("="*60)
    
    # 找出最佳性能
    best_f1 = max(all_results, key=lambda x: x['avg_metrics']['avg_f1_score'])
    best_precision = max(all_results, key=lambda x: x['avg_metrics']['avg_precision'])
    best_recall = max(all_results, key=lambda x: x['avg_metrics']['avg_recall'])
    
    print(f"\n🏆 最佳F1分数: {best_f1['avg_metrics']['avg_f1_score']:.4f}")
    print(f"   实验: {best_f1.get('experiment_id', '未知')}")
    print(f"   模型: {best_f1.get('model_name', '未知')}")
    print(f"   Prompt: {best_f1.get('prompt_type', '未知')}")
    
    print(f"\n📊 性能统计:")
    f1_scores = [r['avg_metrics']['avg_f1_score'] for r in all_results]
    print(f"   F1分数范围: {min(f1_scores):.4f} - {max(f1_scores):.4f}")
    print(f"   F1分数均值: {np.mean(f1_scores):.4f}")
    print(f"   F1分数标准差: {np.std(f1_scores):.4f}")
    
    print(f"\n💡 改进建议:")
    if len(set(r.get('prompt_type') for r in all_results)) > 1:
        prompt_performance = {}
        for result in all_results:
            prompt = result.get('prompt_type', '未知')
            if prompt not in prompt_performance:
                prompt_performance[prompt] = []
            prompt_performance[prompt].append(result['avg_metrics']['avg_f1_score'])
        
        best_prompt = max(prompt_performance.keys(), 
                         key=lambda k: np.mean(prompt_performance[k]))
        worst_prompt = min(prompt_performance.keys(), 
                          key=lambda k: np.mean(prompt_performance[k]))
        
        improvement = (np.mean(prompt_performance[best_prompt]) - 
                      np.mean(prompt_performance[worst_prompt])) / np.mean(prompt_performance[worst_prompt]) * 100
        
        print(f"   1. 使用'{best_prompt}'比'{worst_prompt}'可提升性能{improvement:.1f}%")
    
    print("="*60)