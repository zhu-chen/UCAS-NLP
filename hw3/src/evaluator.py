def calculate_metrics(standard_segmentation: list[str], model_segmentation: list[str]) -> dict:
    if not standard_segmentation and not model_segmentation:
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "correct_segments": 0, "standard_total": 0, "model_total": 0}
    if not standard_segmentation: 
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "correct_segments": 0, "standard_total": 0, "model_total": len(model_segmentation)}
    if not model_segmentation: 
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "correct_segments": 0, "standard_total": len(standard_segmentation), "model_total": 0}

    # 去除空白分词

    standard_segmentation = [word for word in standard_segmentation if word.strip() != '']
    model_segmentation = [word for word in model_segmentation if word.strip() != '']
    
    # 将分词结果转换为词语的起始和结束位置集合
    
    def get_segment_spans(words: list[str]) -> set[tuple[int, int]]:
        spans = set()
        current_pos = 0
        for word in words:
            spans.add((current_pos, current_pos + len(word)))
            current_pos += len(word)
        return spans

    standard_spans = get_segment_spans(standard_segmentation)
    model_spans = get_segment_spans(model_segmentation)

    correct_segments = len(standard_spans.intersection(model_spans))
    
    standard_total = len(standard_spans)
    model_total = len(model_spans)

    precision = correct_segments / model_total if model_total > 0 else 0.0
    recall = correct_segments / standard_total if standard_total > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "correct_segments": correct_segments,
        "standard_total": standard_total,
        "model_total": model_total
    }

if __name__ == '__main__':
    # 使用示例
    std_seg1 = ["我", "爱", "北京", "天安门"]
    model_seg1 = ["我", "爱", "北京天安门"]
    metrics1 = calculate_metrics(std_seg1, model_seg1)
    print(f"标准: {std_seg1}, 模型: {model_seg1} -> 指标: {metrics1}")

    std_seg2 = ["中国", "人民", "从此", "站立", "起来", "了"]
    model_seg2 = ["中国", "人民", "从此", "站立", "起来", "了"]
    metrics2 = calculate_metrics(std_seg2, model_seg2)
    print(f"标准: {std_seg2}, 模型: {model_seg2} -> 指标: {metrics2}")

    std_seg3 = ["他", "说", "的", "确实", "在理"]
    model_seg3 = ["他说", "的", "确实", "在理"]
    metrics3 = calculate_metrics(std_seg3, model_seg3)
    print(f"标准: {std_seg3}, 模型: {model_seg3} -> 指标: {metrics3}")

    std_seg4 = ["欢迎", "新", "同学"]
    model_seg4 = ["欢迎", "新同学"]
    metrics4 = calculate_metrics(std_seg4, model_seg4)
    print(f"标准: {std_seg4}, 模型: {model_seg4} -> 指标: {metrics4}")

    std_seg5 = ["商品", "和", "服务"]
    model_seg5 = ["商品", "和服", "务"]
    metrics5 = calculate_metrics(std_seg5, model_seg5)
    print(f"标准: {std_seg5}, 模型: {model_seg5} -> 指标: {metrics5}")
