import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import argparse

def calculate_entropy(text):
    # 计算文本熵
    counter = Counter(text)
    length = len(text)
    probabilities = [count / length for count in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy

def analyze_text_entropy(file_path, step_size_mb=2, output_dir="results"):
    # 读取文本文件
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # 清除换行符和多余空格
    text = text.replace('\n', ' ').replace('\r', '').strip()

    os.makedirs(output_dir, exist_ok=True)
    
    # 将步长转换为整数，避免 range() 函数的浮点数错误
    step_size = int(step_size_mb * 1024 * 1024)
    
    # 计算不同文本量下的熵
    results = []
    for i in range(step_size, len(text) + 1, step_size):
        sample = text[:i]
        size_mb = i / (1024 * 1024)
        entropy = calculate_entropy(sample)
        results.append({'Size (MB)': round(size_mb, 2), 'Entropy (bits)': round(entropy, 4)})
        
    # 如果最后一个样本不是整数倍step_size
    if len(text) % step_size != 0:
        sample = text
        size_mb = len(text) / (1024 * 1024)
        entropy = calculate_entropy(sample)
        results.append({'Size (MB)': round(size_mb, 2), 'Entropy (bits)': round(entropy, 4)})
    
    # 创建表格
    df = pd.DataFrame(results)
    table_path = os.path.join(output_dir, "entropy_table.csv")
    df.to_csv(table_path, index=False)
    
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(df['Size (MB)'], df['Entropy (bits)'], marker='o')
    plt.title('Text Entropy vs. Text Size')
    plt.xlabel('Text Size (MB)')
    plt.ylabel('Entropy (bits)')
    plt.grid(True)
    
    # 保存图表
    chart_path = os.path.join(output_dir, "entropy_chart.png")
    plt.savefig(chart_path)
    plt.close()
    
    return table_path, chart_path

def main():
    parser = argparse.ArgumentParser(description='Calculate text entropy with increasing text size')
    parser.add_argument('file_path', help='Path to the text file')
    parser.add_argument('--step', type=float, default=2.0, help='Step size in MB')
    parser.add_argument('--output', default='results', help='Output directory')
    
    args = parser.parse_args()
    
    table_path, chart_path = analyze_text_entropy(
        args.file_path, 
        step_size_mb=args.step,
        output_dir=args.output
    )
    
    print(f"Results saved to: {args.output}")
    print(f"Table: {table_path}")
    print(f"Chart: {chart_path}")

if __name__ == "__main__":
    main()
