import torch
import pandas as pd
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# 设置环境参数
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个GPU

# 全局变量存储模型和tokenizer
deepseek_model = None
deepseek_tokenizer = None

# 处理DeepSeek输出，移除思考链
def clean_deepseek_output(text):
    """
    处理DeepSeek输出，移除思考链内容
    
    Args:
        text: DeepSeek模型的输出文本
        
    Returns:
        str: 清理后的文本，不包含思考链
    """
    if not text:
        return text
        
    # 移除思考链内容
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    
    return text

# 加载DeepSeek模型
def load_deepseek_model():
    global deepseek_model, deepseek_tokenizer
    
    # 如果模型已加载，则直接返回
    if deepseek_model is not None and deepseek_tokenizer is not None:
        return deepseek_model, deepseek_tokenizer
        
    print("正在加载DeepSeek模型...")
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("DeepSeek模型加载完成")
    
    # 存储到全局变量
    deepseek_model = model
    deepseek_tokenizer = tokenizer
    
    return model, tokenizer

# 从CSV文件读取文本
def read_text_from_csv(csv_file_path, text_column, row_index=0):
    print(f"从CSV文件读取文本: {csv_file_path}, 行索引: {row_index}")
    try:
        df = pd.read_csv(csv_file_path)
        if text_column not in df.columns:
            raise ValueError(f"CSV文件中不存在列名: {text_column}")
        
        # 检查行索引是否有效
        if row_index >= len(df):
            raise ValueError(f"行索引 {row_index} 超出了CSV文件的行数 {len(df)}")
        
        # 返回指定行的文本
        text = df[text_column].iloc[row_index]
        return text
    except Exception as e:
        print(f"读取CSV文件时出错: {str(e)}")
        return None

# 使用DeepSeek模型进行文本处理
def process_with_deepseek(text, prompt_template=None, max_tokens=512, temperature=0.7, top_p=0.9):
    """
    使用DeepSeek模型处理文本，根据给定的prompt模板
    
    Args:
        text: 输入文本
        prompt_template: 提示模板，如果为None则直接使用text作为输入
        max_tokens: 最大生成令牌数
        temperature: 生成温度
        top_p: 生成top_p值
        
    Returns:
        str: 模型生成的结果
    """
    # 确保模型已加载
    model, tokenizer = load_deepseek_model()
    
    print("正在使用DeepSeek模型处理文本...")
    
    # 准备输入
    if prompt_template:
        # 将文本插入到提示模板中
        prompt = prompt_template.format(text=text)
    else:
        # 如果没有提供模板，直接使用文本
        prompt = text
    
    # 生成结果
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.inference_mode():
        outputs = model.generate(
            inputs=inputs.input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
    
    # 解码结果，只返回新生成的部分
    result = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # 清理输出，移除思考链
    result = clean_deepseek_output(result)
    
    return result

# 使用DeepSeek模型进行文本总结 (保持向后兼容)
def summarize_text(text, custom_prompt=None):
    """
    使用DeepSeek模型总结文本
    
    Args:
        text: 输入文本
        custom_prompt: 自定义提示模板，如果为None则使用默认总结模板
        
    Returns:
        str: 文本总结结果
    """
    # 默认总结提示模板
    default_summary_prompt = """请对以下文本进行总结，提取其中的关键信息：

{text}

总结："""
    
    # 使用自定义提示或默认提示
    prompt_template = custom_prompt if custom_prompt else default_summary_prompt
    
    result = process_with_deepseek(text, prompt_template, max_tokens=512)
    
    # 清理输出，移除思考链
    result = clean_deepseek_output(result)
    
    return result

# 使用DeepSeek比较两段文本的异同
def compare_texts(text1, text2, text1_name="文本1", text2_name="文本2", custom_prompt=None):
    """
    使用DeepSeek模型比较两段文本
    
    Args:
        text1: 第一段文本
        text2: 第二段文本
        text1_name: 第一段文本的名称
        text2_name: 第二段文本的名称
        custom_prompt: 自定义提示模板，如果为None则使用默认比较模板
        
    Returns:
        str: 比较结果
    """
    # 默认比较提示模板
    default_compare_prompt = """请详细比较以下两段文本的异同点：

{text1_name}:
{text1}

{text2_name}:
{text2}

请从以下几个方面进行分析：
1. 两段文本的主题是否一致
2. 两段文本包含哪些相同的关键信息
3. 两段文本各自独有的信息是什么
4. 综合评价两段文本的相似度和差异性

分析结果："""
    
    # 构建完整的文本
    combined_text = {
        "text1": text1,
        "text2": text2,
        "text1_name": text1_name,
        "text2_name": text2_name
    }
    
    # 使用自定义提示或默认提示
    if custom_prompt:
        prompt = custom_prompt.format(**combined_text)
    else:
        prompt = default_compare_prompt.format(**combined_text)
    
    # 直接传递完整的提示，因为文本已经合并到提示中
    result = process_with_deepseek(prompt, None, max_tokens=1024)
    
    # 清理输出，移除思考链
    result = clean_deepseek_output(result)
    
    return result

# 进行文本评分
def rate_text(text, criteria_prompt):
    """
    根据给定标准对文本进行评分
    
    Args:
        text: 要评分的文本
        criteria_prompt: 评分标准提示
        
    Returns:
        str: 评分结果
    """
    result = process_with_deepseek(text, criteria_prompt, max_tokens=512)
    
    # 清理输出，移除思考链
    result = clean_deepseek_output(result)
    
    return result

# 当作为脚本直接运行时执行
if __name__ == "__main__":
    # 设置参数
    csv_file_path = "/home/qiangminc/codes/Shuri/relevance_eval/dataset/text/descriptions.csv"  # 请替换为实际的CSV文件路径
    text_column = "description"  # 请替换为实际的文本列名
    row_index = 0  # 要处理的行索引
    
    # 1. 加载DeepSeek模型
    load_deepseek_model()
    
    # 2. 从CSV读取文本
    text = read_text_from_csv(csv_file_path, text_column, row_index)
    if text is None:
        print("错误: 未能读取文本，程序结束")
        sys.exit(1)
    
    # 3. 使用DeepSeek模型进行文本总结
    summary = summarize_text(text)
    print("\n文本总结结果:")
    print("="*50)
    print(summary)
    print("="*50)
    
    # 4. 使用自定义提示进行文本评分（示例）
    scoring_prompt = """请对以下文本进行评分，评分标准如下：
1. 清晰度：文本表达是否清晰（1-10分）
2. 相关性：文本内容与主题是否相关（1-10分）
3. 完整性：文本是否提供了完整信息（1-10分）
4. 总体评价：总体质量评分（1-10分）

文本：
{text}

评分结果（请给出分数和简要说明）："""
    
    score_result = rate_text(text, scoring_prompt)
    print("\n文本评分结果:")
    print("="*50)
    print(score_result)
    print("="*50)
    
    # 5. 比较文本示例
    text2 = "这是一段测试文本，用于演示比较功能。"
    comparison = compare_texts(text, text2, "原文", "测试文本")
    print("\n文本比较结果:")
    print("="*50)
    print(comparison)
    print("="*50)
