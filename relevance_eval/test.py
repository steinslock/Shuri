import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer

# 全局变量：加载模型（只需加载一次）
# 加载LLaVA模型
llava_model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

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

def analyze_with_deepseek(text_input, prompt_template=None):
    """
    使用DeepSeek模型分析文本输入
    
    参数:
    text_input (str): 需要分析的文本
    prompt_template (str, optional): 提示模板。如果为None，则使用默认模板
    
    返回:
    str: 模型的分析结果
    """
    if prompt_template is None:
        prompt_template = """分析下面这段图像描述，并提供更深入的见解或者修正错误信息：

{text_input}

<think>
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
    deepseek_output = deepseek_tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return deepseek_output

# 演示如何使用这些函数
if __name__ == "__main__":
    # 示例1：使用默认提示
    image_path = "/home/qiangminc/codes/Shuri_eval/relevance_eval/repairing.jpg"
    default_prompt = "Please provide a comprehensive description of this image, covering the following aspects: First, identify and introduce the main building in the image, including its name, geographical location, and historical background. Next, describe the architectural style and exterior features in detail, such as roof design, decorative elements, colors, and materials. Then, describe the surrounding environment, including whether there are other buildings, vegetation, roads, or any notable elements, and assess whether it is under  maintenance. Additionally, analyze the weather conditions at the time the photo was taken, such as lighting and sky conditions. Finally, observe whether there are any prominent people in the image and, if possible, infer their identities or relationships."
    
    # 第一步：获取图像描述
    llava_result = process_image_with_llava(image_path, default_prompt)
    print("LLaVA模型输出：")
    print(llava_result)
    
    # 第二步：分析描述
    deepseek_result = analyze_with_deepseek(llava_result)
    print("\nDeepSeek-R1-Distill-Qwen-14B模型输出：")
    print(deepseek_result)
    
    # 示例2：使用自定义提示（展示如何使用不同的prompt）
    """
    # 自定义图像提示
    custom_image_prompt = "这张图片里有什么？"
    custom_llava_result = process_image_with_llava(image_path, custom_image_prompt)
    
    # 自定义文本分析提示模板
    custom_template = "请用简洁的语言总结以下内容：\n\n{text_input}\n\n总结："
    custom_deepseek_result = analyze_with_deepseek(custom_llava_result, custom_template)
    
    print("\n使用自定义提示的结果：")
    print(custom_deepseek_result)
    """
    