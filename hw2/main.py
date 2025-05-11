import yaml
import torch
import torch.optim as optim
from torch import nn
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
from models import FNN, RNN, LSTM
from utils import load_corpus

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 数据集类
class WordContextDataset(Dataset):
    def __init__(self, corpus, word_to_idx, context_size=2, model_type='FNN'):
        """
        Args:
            corpus: 分词后的语料
            word_to_idx: 词到索引的映射
            context_size: 上下文窗口大小
            model_type: 模型类型，'FNN', 'RNN', 或 'LSTM'
        """
        self.data_pairs = []
        self.model_type = model_type
        
        if model_type == 'FNN':
            # 为FNN创建数据对：(context_words, target_word)
            for sentence in corpus:
                indices = [word_to_idx.get(word, 0) for word in sentence]
                for i in range(context_size, len(indices) - context_size):
                    context = []
                    for j in range(i - context_size, i + context_size + 1):
                        if j != i:  # 忽略当前词
                            context.append(indices[j])
                    
                    target = indices[i]
                    self.data_pairs.append((torch.tensor(context, dtype=torch.long), 
                                          torch.tensor(target, dtype=torch.long)))
        else:  # RNN 或 LSTM
            # 为RNN/LSTM创建数据对：(previous_words, target_word)
            for sentence in corpus:
                indices = [word_to_idx.get(word, 0) for word in sentence]
                for i in range(1, len(indices)):
                    # 获取上下文，最多使用context_size个词
                    context = indices[:i][-context_size:]
                    # 如果上下文长度不足，则在开头填充0
                    if len(context) < context_size:
                        context = [0] * (context_size - len(context)) + context
                    
                    target = indices[i]
                    self.data_pairs.append((torch.tensor(context, dtype=torch.long),
                                          torch.tensor(target, dtype=torch.long)))
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        return self.data_pairs[idx]


# 训练函数
def train_model(model, corpus, word_to_idx, model_type, context_size=2, 
                batch_size=64, epochs=5, learning_rate=0.001):
    """
    训练模型并返回训练后的模型
    
    Args:
        model: 要训练的模型
        corpus: 处理后的语料
        word_to_idx: 词到索引的映射
        model_type: 模型类型，'FNN', 'RNN', 或 'LSTM'
        context_size: 上下文窗口大小
        batch_size: 批处理大小
        epochs: 训练轮数
        learning_rate: 学习率
        
    Returns:
        tuple: (训练后的模型, loss历史记录)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dataset = WordContextDataset(corpus, word_to_idx, context_size, model_type)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 记录loss历史
    loss_history = []
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            context, target = context.to(device), target.to(device)

            optimizer.zero_grad()
            log_probs = model(context)
            loss = criterion(log_probs, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, loss_history


def get_word_vector(model, word_idx, model_type):
    """获取指定词的词向量"""
    return model.embeddings.weight.data[word_idx].cpu().numpy()


def find_similar_words(model, word, word_to_idx, idx_to_word, model_type, top_n=10):
    """
    查找与给定词最相似的词
    
    Args:
        model: 训练好的模型
        word: 目标词
        word_to_idx: 词到索引的映射
        idx_to_word: 索引到词的映射
        model_type: 模型类型
        top_n: 返回结果数量
        
    Returns:
        list: 包含(word, similarity)的列表
    """
    if word not in word_to_idx:
        return []
    
    word_idx = word_to_idx[word]
    word_vector = model.embeddings.weight.data[word_idx].reshape(1, -1)
    
    cos = nn.CosineSimilarity(dim=1)
    similarities = []
    
    for idx, other_word in idx_to_word.items():
        if idx == word_idx:
            continue
        
        other_vector = model.embeddings.weight.data[idx].reshape(1, -1)
        similarity = cos(word_vector, other_vector).item()
        similarities.append((other_word, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_n]


def compare_word_vectors(word, models, model_names, word_to_idx):
    """
    比较不同模型中同一个词的词向量
    
    Args:
        word: 要比较的词
        models: 模型列表
        model_names: 模型名称列表
        word_to_idx: 词到索引的映射
        
    Returns:
        dict: 包含模型间相似度的字典
    """
    if word not in word_to_idx:
        return {}
    
    word_idx = word_to_idx[word]
    vectors = []
    
    for model in models:
        vector = model.embeddings.weight.data[word_idx].cpu().numpy()
        vectors.append(vector)
    
    # 计算模型间词向量的余弦相似度
    results = {}
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            v1 = vectors[i].reshape(1, -1)
            v2 = vectors[j].reshape(1, -1)
            similarity = np.dot(v1, v2.T) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            key = f"{model_names[i]} vs {model_names[j]}"
            results[key] = similarity.item()
    
    return results


def visualize_vectors(models, model_names, words, word_to_idx, output_dir):
    """将不同模型的词向量使用t-SNE降维并可视化"""
    for i, model in enumerate(models):
        # 提取指定词的词向量
        vectors = []
        word_labels = []
        for word in words:
            if word in word_to_idx:
                vectors.append(model.embeddings.weight.data[word_to_idx[word]].cpu().numpy())
                word_labels.append(word)
        
        n_samples = len(vectors)
        if n_samples < 2:  # t-SNE需要至少2个样本
            print(f"警告：模型 {model_names[i]} 的样本数量不足以进行t-SNE降维，跳过可视化。")
            continue
        
        # 动态调整perplexity
        if n_samples <= 10:
            perplexity = min(3, n_samples - 1)
        else:
            perplexity = min(30, n_samples // 3)
        
        print(f"为模型 {model_names[i]} 的 {n_samples} 个样本使用perplexity={perplexity}")
        
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                   learning_rate='auto', init='random')
        vectors_2d = tsne.fit_transform(np.array(vectors))
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c='b', alpha=0.5)
        
        # 添加词语标签
        for j, word in enumerate(word_labels):
            plt.annotate(word, xy=(vectors_2d[j, 0], vectors_2d[j, 1]), 
                        fontproperties='SimHei')  # 使用中文字体
        
        plt.title(f"{model_names[i]} Word Vectors")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_names[i]}_word_vectors.png"))
        plt.close()


def plot_loss_curves(loss_histories, model_names, output_dir):
    """
    绘制不同模型的loss曲线
    
    Args:
        loss_histories: 包含各模型loss历史的字典
        model_names: 模型名称列表
        output_dir: 输出目录
    """
    plt.figure(figsize=(10, 6))
    
    for model_name in model_names:
        loss_history = loss_histories[model_name]
        epochs = range(1, len(loss_history) + 1)
        plt.plot(epochs, loss_history, marker='o', linestyle='-', label=model_name)
    
    plt.title('Training Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    plt.close()
    
    # 将loss数据保存到表格文件
    with open(os.path.join(output_dir, 'loss_history.csv'), 'w') as f:
        # 写入表头
        header = 'Epoch,' + ','.join(model_names)
        f.write(header + '\n')
        
        # 写入数据
        max_epochs = max(len(loss_histories[model]) for model in model_names)
        for epoch in range(max_epochs):
            row = [str(epoch + 1)]
            for model in model_names:
                if epoch < len(loss_histories[model]):
                    row.append(f"{loss_histories[model][epoch]:.4f}")
                else:
                    row.append("")
            f.write(','.join(row) + '\n')


def export_word_vectors(model, model_name, word_to_idx, output_dir):
    """
    将模型的词向量导出到文件
    
    Args:
        model: 训练好的模型
        model_name: 模型名称
        word_to_idx: 词到索引的映射
        output_dir: 输出目录
    """
    # 创建向量子目录
    vectors_dir = os.path.join(output_dir, 'word_vectors')
    os.makedirs(vectors_dir, exist_ok=True)
    
    # 获取所有词向量
    vectors = model.embeddings.weight.data.cpu().numpy()
    
    # 将词向量导出到文本文件
    with open(os.path.join(vectors_dir, f'{model_name}_vectors.txt'), 'w', encoding='utf-8') as f:
        # 写入向量维度信息
        vocab_size, dim = vectors.shape
        f.write(f"{vocab_size} {dim}\n")
        
        # 按词汇表中的顺序写入每个词及其向量
        for word, idx in sorted(word_to_idx.items(), key=lambda x: x[1]):
            vector_str = ' '.join([f"{value:.6f}" for value in vectors[idx]])
            f.write(f"{word} {vector_str}\n")
    
    # 导出为NumPy二进制格式，便于后续加载
    word_list = [word for word, _ in sorted(word_to_idx.items(), key=lambda x: x[1])]
    np.savez(
        os.path.join(vectors_dir, f'{model_name}_vectors.npz'),
        vectors=vectors,
        words=word_list
    )
    
    print(f"Exported {model_name} word vectors to {vectors_dir}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载配置文件
    config_path = os.path.join(current_dir, 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(config.get('random_seed', 42))
    

    # 解析相对路径为绝对路径
    if not os.path.isabs(config['data_path']):
        config['data_path'] = os.path.join(current_dir, config['data_path'])
    if not os.path.isabs(config['output_dir']):
        config['output_dir'] = os.path.join(current_dir, config['output_dir'])

    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 加载语料
    print("Loading corpus...")
    corpus, word_to_idx, idx_to_word = load_corpus(
        config['data_path'], 
        config['vocab_size']
    )
    
    print(f"Corpus loaded. Total sentences: {len(corpus)}")
    print(f"Vocabulary size: {len(word_to_idx)}")
    
    # 初始化模型
    models = {
        'FNN': FNN(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            context_size=config['context_size'],
            hidden_dim=config['hidden_dim']
        ),
        'RNN': RNN(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim']
        ),
        'LSTM': LSTM(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim']
        )
    }
    
    # 训练模型
    trained_models = {}
    loss_histories = {}
    for model_name, model in models.items():
        print(f"\nTraining {model_name} model...")
        trained_model, loss_history = train_model(
            model=model,
            corpus=corpus,
            word_to_idx=word_to_idx,
            model_type=model_name,
            context_size=config['context_size'],
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            learning_rate=config['learning_rate']
        )
        trained_models[model_name] = trained_model
        loss_histories[model_name] = loss_history
        
        # 保存模型
        torch.save(trained_model.state_dict(), 
                   os.path.join(config['output_dir'], f"{model_name}_model.pt"))
        
        # 导出词向量
        export_word_vectors(trained_model, model_name, word_to_idx, config['output_dir'])
    
    # 绘制loss曲线
    print("\nPlotting loss curves...")
    plot_loss_curves(loss_histories, list(models.keys()), config['output_dir'])
    
    # 随机选择20个词
    print("\nSelecting 20 random words...")
    vocab_words = list(word_to_idx.keys())
    if '<UNK>' in vocab_words:
        vocab_words.remove('<UNK>')  # 移除UNK标记
    sample_words = random.sample(vocab_words, min(20, len(vocab_words)))
    
    # 为每个模型计算相似词
    with open(os.path.join(config['output_dir'], 'similar_words.txt'), 'w', encoding='utf-8') as f:
        for model_name, model in trained_models.items():
            f.write(f"\n\n{model_name} Model Similar Words:\n")
            #print(f"\n{model_name} Model Similar Words:")
            
            for word in sample_words:
                similar_words = find_similar_words(
                    model=model,
                    word=word,
                    word_to_idx=word_to_idx,
                    idx_to_word=idx_to_word,
                    model_type=model_name
                )
                
                f.write(f"\nWord: {word}\n")
                #print(f"\nWord: {word}")
                
                for sim_word, similarity in similar_words:
                    f.write(f"  {sim_word}: {similarity:.4f}\n")
                    #print(f"  {sim_word}: {similarity:.4f}")
    
    # 对比不同模型的词向量
    print("\nComparing word vectors across models...")
    with open(os.path.join(config['output_dir'], 'vector_comparison.txt'), 'w', encoding='utf-8') as f:
        for word in sample_words:
            f.write(f"\nWord: {word}\n")
            #print(f"\nWord: {word}")
            
            comparison = compare_word_vectors(
                word=word,
                models=list(trained_models.values()),
                model_names=list(trained_models.keys()),
                word_to_idx=word_to_idx
            )
            
            for models_pair, similarity in comparison.items():
                f.write(f"  {models_pair}: {similarity:.4f}\n")
                #print(f"  {models_pair}: {similarity:.4f}")
    
    # 可视化词向量
    print("\nVisualizing word vectors...")
    visualize_vectors(
        models=list(trained_models.values()),
        model_names=list(trained_models.keys()),
        words=sample_words,
        word_to_idx=word_to_idx,
        output_dir=config['output_dir']
    )
    
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()

