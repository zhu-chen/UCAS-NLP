from .process import count_words

# 语料处理函数
def load_corpus(file_path, vocab_size=1000):
    """
    加载语料并构建词表
    
    Args:
        file_path: 语料文件路径
        vocab_size: 词表大小
        
    Returns:
        corpus: 处理后的语料 (句子列表，每个句子是词语列表)
        word_to_idx: 词到索引的映射
        idx_to_word: 索引到词的映射
    """
    # 使用已有的count_words函数统计词频
    word_counter = count_words(file_path)
    
    # 选择最常见的vocab_size-1个词（预留一个给UNK）
    sorted_words = word_counter.most_common(vocab_size - 1)
    
    # 构建词到索引的映射
    word_to_idx = {'<UNK>': 0}
    idx_to_word = {0: '<UNK>'}
    for i, (word, _) in enumerate(sorted_words, 1):
        word_to_idx[word] = i
        idx_to_word[i] = word
    
    # 构建语料
    corpus = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            # 按空格分割行
            parts = line.strip().split()
            
            # 跳过第一个元素（行ID）
            parts = parts[1:]
            
            sentence = []
            for part in parts:
                # 处理形如 "词/词性" 的格式
                if '/' in part:
                    word, pos = part.rsplit('/', 1)
                    # 忽略标点符号(w)和虚词(u)等
                    if pos not in ['w', 'u' ,'m']:
                        # 如果词不在词表中，则使用<UNK>
                        if word in word_to_idx:
                            sentence.append(word)
                        else:
                            sentence.append('<UNK>')
            
            if len(sentence) > 1:  # 只保留长度大于1的句子
                corpus.append(sentence)
    
    return corpus, word_to_idx, idx_to_word

if __name__ == "__main__":
    # 测试load_corpus函数
    corpus, word_to_idx, idx_to_word = load_corpus('hw2\data\ChineseCorpus199801.txt')
    print("词表大小:", len(word_to_idx))
    print("前10个词:", list(word_to_idx.keys())[:10])
    print("前10个索引:", list(idx_to_word.keys())[:10])
    print("前5个句子:", corpus[:5])