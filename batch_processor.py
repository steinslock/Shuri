import os
import pandas as pd
import re
import time
import json
import csv
from pathlib import Path
import torch
import sys

# 导入DeepSeek模块
from deepseek_pipeline import load_deepseek_model, process_with_deepseek, read_text_from_csv, summarize_text

# 导入LLaVA模块
from Llava import get_llava_description, load_llava_model

# 默认占位符
DEFAULT_SUMMARY_PLACEHOLDER = """
Please summarize the following text in a structured format with bullet points. Ensure that the key points are clear, concise, and well-organized. Only output the summarized content in bullet points.The summary should include:
- A brief overview of the main idea.
- Key details and supporting information.
- Any relevant conclusions or takeaways.
Maintain clarity and coherence while ensuring the summary remains comprehensive. Here is the text:
{text}

"""
DEFAULT_COMPARISON_PLACEHOLDER = """
Compare the following two texts and assess their relevance. Provide a similarity score from 1 to 10, where 1 means completely unrelated and 10 means highly relevant. Output only the score and a brief explanation in bullet points. Do not include any additional text or commentary.

Text 1: {text_summary}
Text 2: {image_description}
"""

# 获取LLaVA的图像描述prompt
def get_llava_prompt():
    """获取LLaVA模型使用的图像描述prompt"""
    try:
        from Llava import DEFAULT_IMAGE_DESCRIPTION_PROMPT
        return DEFAULT_IMAGE_DESCRIPTION_PROMPT
    except ImportError:
        return "请详细描述这张图片中的内容。"

# 获取文件名中的数字
def get_number_from_filename(filename):
    """从文件名中提取数字部分"""
    # 使用正则表达式匹配文件名中的数字部分
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0  # 如果没有找到数字，返回0

# 获取图片文件列表并按数字排序
def get_sorted_image_files(image_dir):
    """获取图片目录中的所有图片文件，并按数字排序"""
    print(f"正在扫描图片目录: {image_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(image_dir):
        print(f"错误: 图片目录不存在: {image_dir}")
        return []
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']
    
    # 获取所有图片文件
    image_files = []
    for file in os.listdir(image_dir):
        file_path = os.path.join(image_dir, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(file)
    
    # 按照文件名中的数字排序
    image_files.sort(key=get_number_from_filename)
    
    print(f"找到 {len(image_files)} 个图片文件")
    for i, file in enumerate(image_files[:5]):
        print(f"  {i+1}. {file}")
    if len(image_files) > 5:
        print(f"  ... 还有 {len(image_files)-5} 个文件")
    
    return image_files

# 文本总结功能
def summarize_text_with_prompt(text, summary_prompt):
    """
    使用指定的提示模板来总结文本
    
    Args:
        text: 要总结的文本
        summary_prompt: 总结提示模板
        
    Returns:
        tuple: (总结结果, 填充后的prompt)
    """
    # 准备填充后的prompt
    try:
        filled_prompt = summary_prompt.format(text=text)
    except:
        # 如果格式化失败，使用简单拼接
        filled_prompt = summary_prompt + "\n\n" + text
    
    # 直接将文本和提示模板分别传递给process_with_deepseek
    result = process_with_deepseek(
        text=text, 
        prompt_template=summary_prompt, 
        max_tokens=512
    )
    
    return result, filled_prompt

# 比较和评分功能
def compare_and_rate(text_summary, image_description, comparison_prompt):
    """
    比较文本总结和图像描述并进行评分
    
    Args:
        text_summary: 文本总结结果
        image_description: 图像描述
        comparison_prompt: 比较和评分的提示模板
        
    Returns:
        str: 比较和评分结果
    """
    # 准备输入上下文
    context = {
        "text_summary": text_summary,
        "image_description": image_description
    }
    
    # 尝试使用提供的键名填充模板
    try:
        filled_text = comparison_prompt.format(**context)
    except KeyError:
        # 尝试使用不同的键名
        context = {
            "text": text_summary,
            "image_desc": image_description
        }
        
        try:
            filled_text = comparison_prompt.format(**context)
        except Exception:
            # 使用简单的拼接作为后备方案
            filled_text = "比较以下两段文本并评分:\n\n"
            filled_text += "文本总结:\n" + text_summary + "\n\n"
            filled_text += "图像描述:\n" + image_description + "\n\n"
            filled_text += "请给出相似度评分(1-10分)并简要解释:"
    
    # 将填充好的文本作为text参数传递，prompt_template设为None
    try:
        result = process_with_deepseek(
            text=filled_text, 
            prompt_template=None, 
            max_tokens=1024
        )
        
        if not result or result.strip() == "":
            result = "相似度评分: 5分\n\n理由:\n- 无法确定具体相似度，给出中等评分"
            
        return result, filled_text  # 返回结果和填充后的prompt
    except Exception:
        # 返回一个默认结果而不是None
        return "相似度评分: 5分\n\n理由:\n- 处理过程中出错，给出默认评分", filled_text

# 批量处理图片和文本
def batch_process(image_dir, csv_file, text_column, output_dir, 
                  summary_prompt=None, comparison_prompt=None,
                  start_idx=0, end_idx=None, max_items=None,
                  save_prompts_to_files=False, output_format="csv"):
    """
    批量处理图片和对应的文本，并保存结果
    
    Args:
        image_dir: 图片目录
        csv_file: CSV文件路径
        text_column: 文本列名
        output_dir: 输出目录
        summary_prompt: 文本总结提示模板
        comparison_prompt: 比较和评分提示模板
        start_idx: 开始处理的索引
        end_idx: 结束处理的索引
        max_items: 最大处理项数
        save_prompts_to_files: 是否将prompt保存到单独的文本文件中
        output_format: 输出格式，可选值为"csv"或"json"
    """
    # 使用默认提示或自定义提示
    summary_prompt = summary_prompt or DEFAULT_SUMMARY_PLACEHOLDER
    comparison_prompt = comparison_prompt or DEFAULT_COMPARISON_PLACEHOLDER
    
    # 获取LLaVA的图像描述prompt
    llava_prompt = get_llava_prompt()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果需要将prompt保存到文件，创建prompt目录
    prompts_dir = None
    if save_prompts_to_files:
        prompts_dir = os.path.join(output_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        
        # 保存默认prompt到文件
        with open(os.path.join(prompts_dir, "llava_default_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(llava_prompt)
        with open(os.path.join(prompts_dir, "summary_default_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(summary_prompt)
        with open(os.path.join(prompts_dir, "comparison_default_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(comparison_prompt)
    
    # 获取排序后的图片文件列表
    image_files = get_sorted_image_files(image_dir)
    if not image_files:
        print("没有找到图片文件，程序结束")
        return
    
    # 加载CSV文件
    try:
        df = pd.read_csv(csv_file)
        print(f"成功加载CSV文件: {csv_file}，共有 {len(df)} 行")
        
        if text_column not in df.columns:
            print(f"错误: CSV文件中不存在列名: {text_column}")
            return
    except Exception as e:
        print(f"加载CSV文件出错: {str(e)}")
        return
    
    # 检查图片数量与CSV行数是否匹配
    if len(image_files) != len(df):
        print(f"警告: 图片数量 ({len(image_files)}) 与CSV行数 ({len(df)}) 不匹配")
    
    # 确定处理范围
    if end_idx is None:
        end_idx = len(image_files)
    
    if max_items is not None:
        end_idx = min(start_idx + max_items, end_idx)
    
    # 预加载DeepSeek模型
    print("预加载DeepSeek模型...")
    load_deepseek_model()
    
    # 预加载LLaVA模型
    print("预加载LLaVA模型...")
    llava_tokenizer, llava_model, llava_image_processor, llava_context_len = load_llava_model()
    
    # 批量处理
    total_items = end_idx - start_idx
    print(f"开始批量处理，从索引 {start_idx} 到 {end_idx-1}，共 {total_items} 项")
    
    results = []
    start_time = time.time()
    
    # 准备输出文件
    csv_output_file = None
    json_output_file = None
    
    if output_format == "csv":
        csv_output_file = os.path.join(output_dir, "results.csv")
        csv_fields = ['index', 'image_file', 'original_text', 'image_description', 'text_summary', 
                     'comparison_result', 'timestamp', 'llava_prompt', 'summary_prompt', 'comparison_prompt']

        csvfile = open(csv_output_file, 'w', newline='', encoding='utf-8-sig')
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
    else:  # json
        json_output_file = os.path.join(output_dir, "results.json")
    
    try:
        for i in range(start_idx, end_idx):
            if i >= len(image_files) or i >= len(df):
                print(f"索引 {i} 超出范围，停止处理")
                break
            
            image_file = image_files[i]
            image_path = os.path.join(image_dir, image_file)
            
            # 显示处理进度
            progress = (i - start_idx + 1) / total_items * 100
            elapsed_time = time.time() - start_time
            avg_time_per_item = elapsed_time / (i - start_idx + 1)
            remaining_time = avg_time_per_item * (end_idx - i - 1)
            
            print(f"\n--- 处理项 {i+1}/{end_idx} ({progress:.1f}%) - 图片: {image_file} ---")
            print(f"已用时间: {elapsed_time/60:.1f}分钟, 预计剩余: {remaining_time/60:.1f}分钟")
            
            # 读取文本
            try:
                # 1. 读取CSV对应行的文本
                text = read_text_from_csv(csv_file, text_column, i)
                if text is None:
                    print(f"警告: 索引 {i} 的文本读取失败，跳过此项")
                    continue
            except Exception as e:
                print(f"读取文本时出错: {str(e)}")
                continue
            
            # 处理图像 - 使用LLaVA模型
            try:
                # 2. 使用LLaVA描述图像
                print(f"开始处理图像: {image_path}")
                image_description = get_llava_description(
                    img_path=image_path,
                    tokenizer=llava_tokenizer,
                    model=llava_model,
                    image_processor=llava_image_processor,
                    context_len=llava_context_len
                )
                if image_description:
                    print(f"成功生成图像描述，长度: {len(image_description)} 字符")
                    # 清理可能导致CSV问题的字符（仅在CSV输出时需要）
                    if output_format == "csv":
                        image_description = image_description.replace('\r', ' ').replace('\n', ' ')
                else:
                    print("警告: 图像描述为空")
            except Exception as e:
                print(f"处理图像时出错: {str(e)}")
                image_description = None
                continue  # 如果图像处理失败，跳过此项
            
            # 文本总结
            try:
                if text:
                    # 3. 使用DeepSeek总结文本
                    print(f"开始总结文本，长度: {len(text)} 字符")
                    # 准备填充后的summary_prompt
                    text_summary, filled_summary_prompt = summarize_text_with_prompt(text, summary_prompt)
                    if text_summary:
                        print(f"成功生成文本总结，长度: {len(text_summary)} 字符")
                        # 清理可能导致CSV问题的字符（仅在CSV输出时需要）
                        if output_format == "csv":
                            text_summary = text_summary.replace('\r', ' ').replace('\n', ' ')
                            filled_summary_prompt = filled_summary_prompt.replace('\r', ' ').replace('\n', ' ')
                    else:
                        print("警告: 文本总结为空")
                        text_summary = None
                else:
                    print("警告: 文本为空，无法生成总结")
                    text_summary = None
                    filled_summary_prompt = ""
            except Exception as e:
                print(f"总结文本时出错: {str(e)}")
                text_summary = None
                filled_summary_prompt = ""
            
            # 比较文本与图像的关系并评分
            filled_comparison_prompt = ""
            try:
                if text_summary and image_description:
                    # 4. 使用DeepSeek比较文本总结和图像描述并评分
                    print(f"开始比较文本总结和图像描述")
                    comparison_result, filled_comparison_prompt = compare_and_rate(text_summary, image_description, comparison_prompt)
                    
                    if comparison_result:
                        print(f"成功生成比较和评分结果，长度: {len(comparison_result)} 字符")
                        # 清理可能导致CSV问题的字符（仅在CSV输出时需要）
                        if output_format == "csv":
                            comparison_result = comparison_result.replace('\r', ' ').replace('\n', ' ')
                            filled_comparison_prompt = filled_comparison_prompt.replace('\r', ' ').replace('\n', ' ')
                    else:
                        print("警告: 比较和评分结果为空")
                        comparison_result = None
                else:
                    print(f"警告: 由于文本总结或图像描述为空，无法生成比较和评分结果")
                    comparison_result = None
            except Exception as e:
                print(f"比较和评分时出错: {str(e)}")
                comparison_result = None
            
            # 收集结果
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # 清理LLaVA prompt中可能导致CSV问题的字符（仅在CSV输出时需要）
            clean_llava_prompt = llava_prompt
            if output_format == "csv":
                clean_llava_prompt = llava_prompt.replace('\r', ' ').replace('\n', ' ')
            
            # 如果需要将prompt保存到文件
            prompt_file_paths = {}
            if save_prompts_to_files and prompts_dir:
                # 为每个项目创建一个子目录
                item_prompts_dir = os.path.join(prompts_dir, f"item_{i}")
                os.makedirs(item_prompts_dir, exist_ok=True)
                
                # 保存各个prompt到文件
                llava_prompt_file = os.path.join(item_prompts_dir, "llava_prompt.txt")
                with open(llava_prompt_file, "w", encoding="utf-8") as f:
                    f.write(llava_prompt)
                prompt_file_paths["llava_prompt"] = llava_prompt_file
                
                if filled_summary_prompt:
                    summary_prompt_file = os.path.join(item_prompts_dir, "summary_prompt.txt")
                    with open(summary_prompt_file, "w", encoding="utf-8") as f:
                        f.write(filled_summary_prompt)
                    prompt_file_paths["summary_prompt"] = summary_prompt_file
                
                if filled_comparison_prompt:
                    comparison_prompt_file = os.path.join(item_prompts_dir, "comparison_prompt.txt")
                    with open(comparison_prompt_file, "w", encoding="utf-8") as f:
                        f.write(filled_comparison_prompt)
                    prompt_file_paths["comparison_prompt"] = comparison_prompt_file
            
            # 准备结果字典
            if output_format == "csv":
                # 对于CSV，需要处理字符串并可能使用文件路径
                item_result = {
                    "index": i,
                    "image_file": image_file,
                    "original_text": text.replace('\r', ' ').replace('\n', ' ') if text else "",
                    "image_description": image_description or "",
                    "text_summary": text_summary or "",
                    "comparison_result": comparison_result or "",
                    "timestamp": timestamp,
                    "llava_prompt": clean_llava_prompt if not save_prompts_to_files else prompt_file_paths.get("llava_prompt", ""),
                    "summary_prompt": filled_summary_prompt or "" if not save_prompts_to_files else prompt_file_paths.get("summary_prompt", ""),
                    "comparison_prompt": filled_comparison_prompt or "" if not save_prompts_to_files else prompt_file_paths.get("comparison_prompt", "")
                }
            else:
                # 对于JSON，可以保留完整的文本内容
                item_result = {
                    "index": i,
                    "image_file": image_file,
                    "original_text": text,
                    "image_description": image_description,
                    "text_summary": text_summary,
                    "comparison_result": comparison_result,
                    "timestamp": timestamp,
                    "prompts": {
                        "llava_prompt": llava_prompt if not save_prompts_to_files else prompt_file_paths.get("llava_prompt", ""),
                        "summary_prompt": filled_summary_prompt if not save_prompts_to_files else prompt_file_paths.get("summary_prompt", ""),
                        "comparison_prompt": filled_comparison_prompt if not save_prompts_to_files else prompt_file_paths.get("comparison_prompt", "")
                    }
                }
            
            results.append(item_result)
            
            # 写入结果
            if output_format == "csv":
                print(f"写入结果到CSV文件: {csv_output_file}")
                try:
                    writer.writerow(item_result)
                    csvfile.flush()  # 确保每次循环后数据都被写入文件
                except Exception as e:
                    print(f"CSV写入失败: {str(e)}")
                    # 尝试使用更安全的方式写入
                    try:
                        safe_item = {}
                        for key, value in item_result.items():
                            if isinstance(value, str):
                                # 移除所有可能导致CSV问题的字符
                                safe_item[key] = ''.join(c for c in value if c.isprintable())
                            else:
                                safe_item[key] = value
                        writer.writerow(safe_item)
                        csvfile.flush()
                        print("使用安全模式重新写入成功")
                    except Exception as e2:
                        print(f"安全模式写入也失败: {str(e2)}")
                
                print(f"结果已写入CSV文件: {csv_output_file}")
            
        # 如果是JSON格式，在处理完所有项目后一次性写入
        if output_format == "json":
            print(f"写入结果到JSON文件: {json_output_file}")
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"结果已保存到JSON文件: {json_output_file}")
    
    finally:
        # 确保CSV文件被关闭
        if output_format == "csv" and csvfile:
            csvfile.close()
    
    # 生成简要统计信息
    total_time = time.time() - start_time
    avg_time = total_time / len(results) if results else 0
    
    stats = {
        "total_items_processed": len(results),
        "total_time_seconds": total_time,
        "average_time_per_item_seconds": avg_time,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_file": csv_output_file if output_format == "csv" else json_output_file,
        "output_format": output_format
    }
    
    stats_file = os.path.join(output_dir, "stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n批量处理完成！总共处理了 {len(results)} 项")
    print(f"结果已保存到{output_format.upper()}文件: {csv_output_file if output_format == 'csv' else json_output_file}")
    print(f"统计信息已保存到: {stats_file}")
    print(f"总用时: {total_time/60:.1f}分钟, 平均每项: {avg_time/60:.1f}分钟")

if __name__ == "__main__":
    # 设置默认参数
    default_image_dir = "/home/qiangminc/codes/Shuri/relevance_eval/dataset/image"
    default_csv_file = "/home/qiangminc/codes/Shuri/relevance_eval/dataset/data.csv"
    default_text_column = "English-version"
    default_output_dir = "/home/qiangminc/codes/Shuri/relevance_eval"
    
    # 从命令行参数获取配置
    import argparse
    parser = argparse.ArgumentParser(description="批量处理图片和文本并进行分析")
    
    parser.add_argument("--image_dir", type=str, default=default_image_dir,
                        help=f"图片目录 (默认: {default_image_dir})")
    parser.add_argument("--csv_file", type=str, default=default_csv_file,
                        help=f"CSV文件路径 (默认: {default_csv_file})")
    parser.add_argument("--text_column", type=str, default=default_text_column,
                        help=f"文本列名 (默认: {default_text_column})")
    parser.add_argument("--output_dir", type=str, default=default_output_dir,
                        help=f"输出目录 (默认: {default_output_dir})")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="开始处理的索引 (默认: 0)")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="结束处理的索引 (默认: 处理到最后)")
    parser.add_argument("--max_items", type=int, default=None,
                        help="最大处理项数 (默认: 无限制)")
    parser.add_argument("--summary_prompt_file", type=str, default=None,
                        help="文本总结提示模板文件 (默认: 使用占位符)")
    parser.add_argument("--comparison_prompt_file", type=str, default=None,
                        help="比较和评分提示模板文件 (默认: 使用占位符)")
    parser.add_argument("--save_prompts_to_files", action="store_true",
                        help="将prompt保存到单独的文本文件中，而不是直接写入CSV (默认: 否)")
    parser.add_argument("--output_format", type=str, choices=["csv", "json"], default="csv",
                        help="输出格式，可选值为'csv'或'json' (默认: csv)")
    
    args = parser.parse_args()
    
    # 读取自定义提示模板
    summary_prompt = DEFAULT_SUMMARY_PLACEHOLDER
    comparison_prompt = DEFAULT_COMPARISON_PLACEHOLDER
    
    # 如果指定了总结提示模板文件，则加载文件内容
    if args.summary_prompt_file:
        try:
            with open(args.summary_prompt_file, 'r', encoding='utf-8') as f:
                summary_prompt = f.read()
            print(f"已加载自定义总结提示模板: {args.summary_prompt_file}")
        except Exception as e:
            print(f"警告: 无法加载总结提示模板文件: {str(e)}")
    
    # 如果指定了比较提示模板文件，则加载文件内容
    if args.comparison_prompt_file:
        try:
            with open(args.comparison_prompt_file, 'r', encoding='utf-8') as f:
                comparison_prompt = f.read()
            print(f"已加载自定义比较和评分提示模板: {args.comparison_prompt_file}")
        except Exception as e:
            print(f"警告: 无法加载比较和评分提示模板文件: {str(e)}")
    
    # 调用批量处理函数
    batch_process(
        args.image_dir,
        args.csv_file,
        args.text_column,
        args.output_dir,
        summary_prompt,
        comparison_prompt,
        args.start_idx,
        args.end_idx,
        args.max_items,
        args.save_prompts_to_files,
        args.output_format
    ) 