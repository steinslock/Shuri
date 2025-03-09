import torch
import os
import sys
from PIL import Image

# 添加LLaVA路径到系统路径
LLAVA_PATH = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA"
sys.path.append(LLAVA_PATH)

# 检查sentencepiece是否正确安装
try:
    import sentencepiece as spm
    print("sentencepiece已成功导入")
except ImportError:
    print("警告: sentencepiece导入失败，尝试再次安装")
    os.system("pip install sentencepiece==0.1.99")
    try:
        import sentencepiece as spm
        print("sentencepiece现在已成功导入")
    except ImportError:
        print("错误: 无法导入sentencepiece，请手动检查安装")
        sys.exit(1)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token

# 设置模型路径
lora_model_path = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA/checkpoints/llava-v.15-7b-lora-正殿"
# 检查LoRA目录中的文件
print(f"检查LoRA目录中的文件:")
if os.path.exists(lora_model_path):
    files = os.listdir(lora_model_path)
    for file in files:
        print(f" - {file}")
else:
    print(f"警告: LoRA目录不存在: {lora_model_path}")

# 设置基础模型 - 根据您的环境配置更改此路径
base_model_path = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA/checkpoints/llava-v1.5-7b"  # 使用Hugging Face上的模型名称或本地路径
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
image_path = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/finetuning_dataset/image/守礼門/3.jpg"
prompt = """You must provide a complete response by strictly following the structured format below. Do not skip any sections. If information is not available, state "Information not available."  

###  Building Information  
- 1. What is the name of this building?  
- 2. Where is it geographically located?  
- 3. Provide a brief historical background, including its construction period and cultural significance.  

###  Architectural Style and Exterior Features  
- 4. What architectural style does this building belong to?  
- 5. Describe the roof design, decorative elements, colors, and materials used in its construction.  

### Surrounding Environment  
- 6. Are there any other buildings nearby? If so, describe them.  
- 7. Is there any vegetation, roads, or other notable environmental features?  
- 8. Is the building currently under maintenance or restoration?  

### Weather and Lighting Conditions  
- 9. What are the weather conditions in this image? (e.g., sunny, cloudy, rainy)  
- 10. What is the direction of sunlight? (e.g., morning, noon, evening)  
- 11. Are there any notable sky features, such as clouds, the sun, or special weather phenomena?  

### People and Activities  
- 12. Are there any prominent people in the image?  
- 13. What might be their identities or occupations? (e.g., tourists, staff, locals)  
- 14. What are they doing? Are their actions related to the building or environment?  

Please answer each question separately in a step-by-step manner, without merging responses.  
"""

# 检查图像是否存在
if os.path.exists(image_path):
    print(f"图像文件存在: {image_path}")
else:
    print(f"错误: 图像文件不存在: {image_path}")
    sys.exit(1)

# 禁用torch初始化以加快加载速度
def disable_torch_init():
    """Disable the redundant torch default initialization to accelerate model creation."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载模型
print("正在加载模型...")
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
    print("模型加载成功!")

    # 加载和处理图像
    print(f"正在加载图像: {image_path}")
    image = Image.open(image_path).convert('RGB')
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
            temperature=0.2,
            top_p=0.7,
            max_new_tokens=2048,
            use_cache=True
        )

    # 解码输出
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 提取回答
    answer = output_text.split(conv.roles[1] + ": ")[-1].strip()
    print("\n" + "="*50)
    print("问题:", prompt)
    print("回答:", answer)
    print("="*50)
    
except Exception as e:
    print(f"错误: {str(e)}")
    import traceback
    traceback.print_exc()
