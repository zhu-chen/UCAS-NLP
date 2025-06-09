import requests
import json
import os
import time
import yaml  # 添加yaml库

def load_api_config():
    """
    从配置文件加载API密钥配置
    """
    config_path = os.path.join("..","configs", "api_config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"加载配置文件出错: {e}")
        return {"siliconflow": {"api_key": "sk-your_default_api_key_here"}}

# 加载API配置
api_config = load_api_config()
SILICONFLOW_API_KEY = api_config.get("api_key", "sk-your_default_api_key_here")

# --- SiliconFlow API 调用示例 ---
def call_siliconflow_llm(prompt: str, sentence: str, model_name: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", max_retries=3, retry_delay=2):
    """
    调用 SiliconFlow 的大模型 API 进行分词。
    prompt: 包含占位符 {sentence} 的提示词模板。
    sentence: 需要分词的句子。
    model_name: 要使用的模型名称。
    """
    
    url = "https://api.siliconflow.cn/v1/chat/completions"
    
    full_prompt_content = prompt.format(sentence=sentence)

    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": full_prompt_content}
        ],
        "stream": False,
        "max_tokens": 128, 
        "temperature": 0.1, 
        "thinking_budget": 1024,
        "response_format": {"type": "text"},
    }
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    # 定义调试信息打印函数
    def print_debug_info():
        """打印详细的调试信息"""
        print("\n" + "="*50)
        print("API调用失败 - 调试信息:")
        print("="*50)
        print(f"URL: {url}")
        print(f"Model: {model_name}")
        print(f"Sentence: {sentence[:100]}...")  # 只显示前100个字符
        print("\nHeaders:")
        for key, value in headers.items():
            print(f"  {key}: {value}")
        print("\nPayload:")
        payload_copy = payload.copy()
        if 'messages' in payload_copy and len(payload_copy['messages']) > 0:
            # 截断过长的消息内容
            if len(payload_copy['messages'][0]['content']) > 200:
                payload_copy['messages'][0]['content'] = payload_copy['messages'][0]['content'][:200] + "..."
        print(json.dumps(payload_copy, ensure_ascii=False, indent=2))
        print("="*50 + "\n")

    for attempt in range(max_retries):
        try:
            print(f"  API调用尝试 {attempt + 1}/{max_retries}...")
            
            # 使用更短的超时时间，分别设置连接和读取超时
            response = requests.post(url, json=payload, headers=headers, 
                                   timeout=(10, 120))  # (连接超时, 读取超时)
            response.raise_for_status()  
            
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0].get("message", {}).get("content", "")
                
                # 清理可能的前缀（如"分词结果："）
                content = content.strip()
                prefixes_to_remove = ["分词结果：", "分词结果:", "分词结果: ", "输出：", "输出:", "输出: "]
                for prefix in prefixes_to_remove:
                    if content.startswith(prefix):
                        content = content[len(prefix):]
                        break
                
                segmented_words = content.strip().split() 
                print(f"  API调用成功，返回 {len(segmented_words)} 个词")
                return segmented_words
            else:
                print(f"API响应格式不符合预期: {response_data}")
                print_debug_info()
                return None
                
        except requests.exceptions.Timeout as e:
            print(f"  SiliconFlow API 超时 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == 0:  # 只在第一次失败时打印调试信息
                print_debug_info()
            if attempt < max_retries - 1:
                print(f"  等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay *= 1.5  # 指数退避策略
            else:
                print("  已达到最大重试次数。")
                return None
                
        except requests.exceptions.ConnectionError as e:
            print(f"  网络连接错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == 0:  # 只在第一次失败时打印调试信息
                print_debug_info()
            if attempt < max_retries - 1:
                print(f"  等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("  网络连接持续失败，请检查网络连接。")
                return None
                
        except requests.exceptions.HTTPError as e:
            print(f"  HTTP错误 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == 0:  # 只在第一次失败时打印调试信息
                print_debug_info()
                # 如果有响应内容，也打印出来
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_content = e.response.text
                        print(f"错误响应内容: {error_content[:500]}...")
                    except:
                        print("无法获取错误响应内容")
            
            if e.response.status_code == 429:  # 速率限制
                print("  遇到速率限制，延长等待时间...")
                time.sleep(retry_delay * 2)
            elif e.response.status_code >= 500:  # 服务器错误
                print("  服务器错误，稍后重试...")
                time.sleep(retry_delay)
            else:
                print(f"  客户端错误，停止重试: {e.response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"  请求异常 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == 0:  # 只在第一次失败时打印调试信息
                print_debug_info()
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("  已达到最大重试次数。")
                return None
                
        except json.JSONDecodeError as e:
            print(f"解析SiliconFlow API响应JSON时出错: {e}")
            print_debug_info()
            if 'response' in locals():
                print(f"原始响应文本: {response.text[:500]}...") 
            return None
            
        except Exception as e:
            print(f"调用SiliconFlow API时发生未知错误: {e}")
            print_debug_info()
            return None
            
    return None

def get_llm_segmentation(prompt_template: str, sentence: str, model_name: str) -> list[str] | None:
    # 原本打算试多个api的,但后来发现没必要
    return call_siliconflow_llm(prompt_template, sentence, model_name)

if __name__ == '__main__':
    # 使用示例
    if not SILICONFLOW_API_KEY or SILICONFLOW_API_KEY == "sk-your_default_api_key_here":
        print("警告: SiliconFlow API Key 未设置或使用的是默认值。API调用可能失败。")
        print("请在 configs/api_config.yaml 文件中设置正确的 API Key")
    
    test_sentence = "我爱北京天安门"
    # 基础提示词模板
    base_prompt = "请对以下句子进行中文分词，分词结果用空格隔开：\n句子：{sentence}\n分词结果："
    # 少样本提示词示例
    few_shot_prompt = (
        "请对以下句子进行中文分词，分词结果用空格隔开。\n"
        "例如：\n"
        "句子：他来到了网易杭研大厦\n"
        "分词结果：他 来到 了 网易 杭研 大厦\n"
        "句子：{sentence}\n"
        "分词结果："
    )

    print(f"\n测试 SiliconFlow API (模型: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B):")
    print(f"句子: {test_sentence}")
    
    print("\n使用基础提示词:")
    result_base = get_llm_segmentation(
        prompt_template=base_prompt,
        sentence=test_sentence,
        model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" 
    )
    if result_base:
        print(f"分词结果 (基础提示词): {result_base}")
    else:
        print("未能获取分词结果 (基础提示词)。")

    print("\n使用少样本提示词:")
    result_few_shot = get_llm_segmentation(
        prompt_template=few_shot_prompt,
        sentence=test_sentence,
        model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    )
    if result_few_shot:
        print(f"分词结果 (少样本提示词): {result_few_shot}")
    else:
        print("未能获取分词结果 (少样本提示词)。")
