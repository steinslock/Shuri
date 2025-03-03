import os
import csv
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

# 全局变量：加载模型（只需加载一次）
# 加载LLaVA模型
llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# 创建两个模版，用于分点总结和比较分析并生成分数
summerize_template = "Please summarize the following passage in a structured format with key points. List the main ideas concisely and clearly. Number each point for clarity：\n\n{text_input}\n\nSummary："
comparison_template = "Please compare the following two passages and provide a detailed analysis of the similarities and differences between them. Explain the reasons for any differences in detail.\n\nPassage 1：\n{text_input1}\n\nPassage 2：\n{text_input2}\n\nAnalysis："

# 加载DeepSeek模型
deepseek_model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", 
    torch_dtype=torch.float16, 
    device_map="auto",
    trust_remote_code=True
)
deepseek_tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    trust_remote_code=True
)

def process_image_with_llava(image_path, prompt_text):
    """
    使用LLaVA模型处理图像并返回描述
    
    参数:
    image_path (str): 图像文件的路径
    prompt_text (str): 发送给模型的提示文本
    返回:
    str: 模型生成的图像描述
    """
    # 加载图像
    image = Image.open(image_path)
    
    # 准备图像提示
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    
    prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # 处理输入
    inputs = llava_processor(image, text=prompt, padding=True, return_tensors="pt").to(llava_model.device, torch.float16)
    
    # 生成描述
    generate_ids = llava_model.generate(**inputs, max_new_tokens=512)
    llava_output = llava_processor.batch_decode(generate_ids, skip_special_tokens=True)
    
    # 提取Assistant的回复
    assistant_response = llava_output[0].split("ASSISTANT:")[-1].strip()
    return assistant_response

def analyze_with_deepseek(text_input, prompt_template):
    """
    使用DeepSeek模型分析文本输入
    
    参数:
    text_input (str): 需要分析的文本
    prompt_template (str): 提示模板。
    
    返回:
    str: 模型的分析结果

    """
    
    # 准备提示
    deepseek_prompt = prompt_template.format(text_input=text_input)
    
    # 编码并生成回复
    inputs = deepseek_tokenizer(deepseek_prompt, return_tensors="pt").to(deepseek_model.device)
    with torch.no_grad():
        output_ids = deepseek_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )
    
    # 解码输出
    deepseek_output_with_thinking = deepseek_tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    deepseek_output_without_thinking = deepseek_output_with_thinking.split("</think>")[-1].strip()

    return deepseek_output_without_thinking         #返回不带思考过程的输出
    # return deepseek_output_with_thinking          #返回带思考过程的输出


# 演示如何使用这些函数
if __name__ == "__main__":
    # 图像和文本文件路径
    image_folder = "/home/qiangminc/codes/Shuri_eval/relevance_eval/dataset/image"
    text_csv_path = "/home/qiangminc/codes/Shuri_eval/relevance_eval/dataset/text_csv.csv"
    
    # 读取CSV文件中的文本
    with open(text_csv_path, newline='', encoding='utf-8') as csvfile:
        text_reader = csv.reader(csvfile)
        text_list = [row[0] for row in text_reader]  # 假设文本在第一列

    # 遍历图像文件夹中的每一张图片
    for idx, image_filename in enumerate(sorted(os.listdir(image_folder))):
        image_path = os.path.join(image_folder, image_filename)
        
        # 确保索引不超出文本列表的范围
        if idx >= len(text_list):
            print("警告：图像数量超过文本数量，停止处理。")
            break
        
        original_comment = text_list[idx]
        
        # 第一步：获取图像描述
        default_prompt = "Please provide a comprehensive description of this image, covering the following aspects: First, identify and introduce the main building in the image, including its name, geographical location, and historical background. Next, describe the architectural style and exterior features in detail, such as roof design, decorative elements, colors, and materials. Then, describe the surrounding environment, including whether there are other buildings, vegetation, roads, or any notable elements, and assess whether it is under maintenance. Additionally, analyze the weather conditions at the time the photo was taken, such as lighting and sky conditions. Finally, observe whether there are any prominent people in the image and, if possible, infer their identities or relationships."
        llava_result = process_image_with_llava(image_path, default_prompt)
        
        # 生成分点总结的prompt
        summary_prompt_generated = generate_summary_prompt(llava_result)
        summary_result = analyze_with_deepseek(llava_result, summary_prompt_generated)
        print(f"\n分点总结结果（图像: {image_filename}）：")
        print(summary_result)
        
        # # 生成比较分析的prompt
        # comparison_prompt_generated = generate_comparison_prompt(llava_result, original_comment)
        # comparison_result = analyze_with_deepseek(None, comparison_prompt_generated)
        # print(f"\n比较分析结果（图像: {image_filename}）：")
        # print(comparison_result)
