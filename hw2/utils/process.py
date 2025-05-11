from collections import Counter
import os

def count_words(filename):
    word_counter = Counter()
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # 跳过空行
            if not line.strip():
                continue
                
            # 按空格分割行
            parts = line.strip().split()
            
            # 跳过第一个元素（行ID）
            parts = parts[1:]
            
            for part in parts:
                # 处理形如 "词/词性" 的格式
                if '/' in part:
                    word, pos = part.rsplit('/', 1)  # 从右边分割，处理词中可能包含/的情况
                    # 忽略标点符号(w)和虚词(u)等
                    if pos not in ['w', 'u','m']:
                        word_counter[word] += 1
    return word_counter
    
     
def save_word_frequency(counter, output_file):

    with open(output_file, 'w', encoding='utf-8') as f:
        # 按词频从大到小排序
        for word, count in counter.most_common():
            f.write(f"{word} {count}\n")

def process(input_file, output_file):    
    # 统计词频
    word_counter = count_words(input_file)
    
    # 保存结果
    save_word_frequency(word_counter, output_file)
    
    print(f"词频统计完成，结果已保存至 {output_file}")
    print(f"共统计了 {len(word_counter)} 个不同的词")

if __name__ == "__main__":
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建输入和输出文件的绝对路径
    input_file = os.path.join(script_dir, '..', 'data', 'ChineseCorpus199801.txt')
    output_file = os.path.join(script_dir, '..', 'data', 'word_frequency.txt')
    process(input_file=input_file, output_file=output_file)
