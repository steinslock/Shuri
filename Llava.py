import torch
import os
import sys
from PIL import Image

# 添加LLaVA路径到系统路径
LLAVA_PATH = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA"
sys.path.append(LLAVA_PATH)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token

# 默认图像描述提示
DEFAULT_IMAGE_DESCRIPTION_PROMPT = """
 Is the main subject in the image a building? (Yes / No / Unclear from the image) If there is a building, answer the following questions. You **must** follow the structured response format exactly as shown below. Do **not** provide additional information beyond what is required. Each response **must be numbered** and correspond to the provided question.

        - 1. Is the building under maintenance? You **must** determine it **only by analyzing the visual elements in the image** and you **must** provide visible evidence from the image that supports your answer.     
        - 2. What are the weather conditions in this image? (e.g., sunny, cloudy, rainy) 
        - 3. Does the image appear to be taken during the day or at night? 
        - 4. What is the name of this building?  
        - 5. Where is it geographically located?  
        - 6. Provide a brief historical background, including its construction period, cultural significance.  
        - 7. What architectural style does this building belong to? Describe the decorative elements, colors, and materials used in its construction.   
        Please answer each question separately in a step-by-step manner, without merging responses.  
"""

# 全局变量存储已加载的模型
llava_tokenizer = None
llava_model = None
llava_image_processor = None
llava_context_len = None

# 禁用torch初始化以加快加载速度
def disable_torch_init():
    """Disable the redundant torch default initialization to accelerate model creation."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# 加载LLaVA模型
def load_llava_model():
    """
    加载LLaVA模型，如果已经加载则直接返回
    
    Returns:
        tuple: (tokenizer, model, image_processor, context_len)
    """
    global llava_tokenizer, llava_model, llava_image_processor, llava_context_len
    
    # 如果模型已加载，直接返回
    if llava_model is not None and llava_tokenizer is not None:
        return llava_tokenizer, llava_model, llava_image_processor, llava_context_len
    
    # 设置模型路径
    lora_model_path = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA/checkpoints/llava-v.15-7b-lora-综合2"
    base_model_path = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA/checkpoints/llava-v1.5-7b"
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载LLaVA模型...")
    disable_torch_init()
    
    try:
        # 使用原始模型加载LoRA权重
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=lora_model_path,
            model_base=base_model_path,
            model_name="llava-v1.5-7b-lora",
            load_8bit=False,
            load_4bit=False,
            device_map=device
        )
        print("LLaVA模型加载成功!")
        
        # 存储到全局变量
        llava_tokenizer = tokenizer
        llava_model = model
        llava_image_processor = image_processor
        llava_context_len = context_len
        
        return tokenizer, model, image_processor, context_len
    except Exception as e:
        print(f"加载LLaVA模型出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# 提供一个函数用于被其他模块调用
def get_llava_description(img_path, tokenizer=None, model=None, image_processor=None, context_len=None, custom_prompt=None):
    """
    使用LLaVA模型分析图像并返回描述
    
    Args:
        img_path: 图像文件路径
        tokenizer: 预加载的tokenizer，如果为None则加载新的
        model: 预加载的模型，如果为None则加载新的
        image_processor: 预加载的图像处理器，如果为None则加载新的
        context_len: 预加载的上下文长度，如果为None则加载新的
        custom_prompt: 自定义提示文本，如果为None则使用默认提示
        
    Returns:
        str: 模型生成的图像描述
    """
    # 检查图像是否存在
    if not os.path.exists(img_path):
        print(f"错误: 图像文件不存在: {img_path}")
        return None
    
    # 使用提供的提示文本或默认提示
    if custom_prompt is not None:
        prompt = custom_prompt
    else:
        prompt = DEFAULT_IMAGE_DESCRIPTION_PROMPT
    
    # 如果没有提供预加载的模型，则加载模型
    if model is None or tokenizer is None or image_processor is None or context_len is None:
        print("加载LLaVA模型...")
        tokenizer, model, image_processor, context_len = load_llava_model()
        if model is None or tokenizer is None:
            print("错误: 无法加载LLaVA模型")
            return None
    else:
        print("使用预加载的LLaVA模型...")
    
    # 设置设备
    device = next(model.parameters()).device
    
    try:
        # 加载和处理图像
        print(f"正在处理图像: {img_path}")
        image = Image.open(img_path).convert('RGB')
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(device)

        # 设置对话模板
        conv_mode = "llava_v1"
        conv = conv_templates[conv_mode].copy()

        # 添加提示
        inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # 转换输入为模型可接受的格式
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        # 进行推理
        print("正在进行推理...")
        with torch.inference_mode():
            outputs = model.generate(
                inputs=input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.1,
                top_p=0.7,
                max_new_tokens=2048,
                use_cache=True
            )

        # 解码输出
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取回答
        result = output_text.split(conv.roles[1] + ": ")[-1].strip()
        return result
        
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 设置模型路径
lora_model_path = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA/checkpoints/llava-v.15-7b-lora-综合2"
# 检查LoRA目录中的文件
print(f"检查LoRA目录中的文件:")
if os.path.exists(lora_model_path):
    files = os.listdir(lora_model_path)
    for file in files:
        print(f" - {file}")
else:
    print(f"警告: LoRA目录不存在: {lora_model_path}")

# 设置基础模型 - 根据您的环境配置更改此路径
base_model_path = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA/checkpoints/llava-v1.5-7b"  
# 检查基础模型目录中的文件
print(f"检查基础模型目录中的文件:")
if os.path.exists(base_model_path):
    files = os.listdir(base_model_path)
    print(f" - 发现{len(files)}个文件")
    # 只显示前10个文件避免输出过多
    for file in files[:10]:
        print(f" - {file}")
    if len(files) > 10:
        print(f" - ... 还有{len(files)-10}个文件")
else:
    print(f"警告: 基础模型目录不存在: {base_model_path}")

# 设置图像路径和提示
image_path = "/home/qiangminc/codes/Shuri/relevance_eval/dataset/image/3.jpg"
prompt = """
Is the main subject in the image a building? (Yes / No / Unclear from the image) If there is a building, answer the following questions. You **must** follow the structured response format exactly as shown below. Do **not** provide additional information beyond what is required. Each response **must be numbered** and correspond to the provided question.

- 1. Is the building under maintenance? You **must** determine it **only by analyzing the visual elements in the image** and you **must** provide visible evidence from the image that supports your answer.     
- 2. What are the weather conditions in this image? (e.g., sunny, cloudy, rainy) 
- 3. Does the image appear to be taken during the day or at night? 
- 4. What is the name of this building?  
- 5. Where is it geographically located?  
- 6. Provide a brief historical background, including its construction period, cultural significance.  
- 7. What architectural style does this building belong to? Describe the decorative elements, colors, and materials used in its construction.   
Please answer each question separately in a step-by-step manner, without merging responses.  

"""

# 检查图像是否存在
if os.path.exists(image_path):
    print(f"图像文件存在: {image_path}")
else:
    print(f"错误: 图像文件不存在: {image_path}")
    sys.exit(1)

# 当作为脚本直接运行时执行
if __name__ == "__main__":
    # 预加载LLaVA模型
    load_llava_model()
    
    try:
        # 使用get_llava_description函数处理图像
        answer = get_llava_description(image_path, prompt)
        
        print("\n" + "="*50)
        print("问题:", prompt)
        print("回答:", answer)
        print("="*50)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
