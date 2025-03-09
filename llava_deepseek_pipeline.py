import torch
import argparse
import json
from PIL import Image
from transformers import TextStreamer

# 导入LLaVA官方库中的函数
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

def load_llava_official(model_path, model_base=None, load_8bit=False, load_4bit=True, device="cuda"):
    """使用官方方法加载LLaVA模型"""
    disable_torch_init()
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_8bit, load_4bit, device=device
    )
    
    # 确定对话模式
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
        
    return model, tokenizer, image_processor, conv_mode

def process_with_llava_official(model, tokenizer, image_processor, conv_mode, image_path, prompt, max_new_tokens=200):
    """使用官方方法处理图像并获取LLaVA输出"""
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    image_size = image.size
    
    # 处理图像
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    
    # 创建对话模板
    conv = conv_templates[conv_mode].copy()
    
    # 添加图像标记
    if model.config.mm_use_im_start_end:
        prompt_with_image = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    else:
        prompt_with_image = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    # 准备对话
    conv.append_message(conv.roles[0], prompt_with_image)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()
    
    # 编码输入
    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    
    # 准备生成参数
    stop_str = conv.sep if conv.sep_style != conv_templates["mpt"].sep_style else conv.sep2
    
    # 生成回复
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=[image_size],
            do_sample=True,
            temperature=0.1,
            max_new_tokens=max_new_tokens,
            use_cache=True
        )
    
    # 解码输出
    outputs = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    
    # 提取回答部分
    assistant_response = outputs.split(conv.roles[1] + ": ")[-1].strip()
    return assistant_response

def parse_args():
    parser = argparse.ArgumentParser(description="LLaVA推理管道")
    parser.add_argument("--llava_model_path", type=str, default="/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/LLaVA/checkpoints/llava-v1.5-7b-lora-merged-正殿", help="微调后的LLaVA模型路径")
    parser.add_argument("--llava_model_base", type=str, default=None, help="LLaVA基础模型路径")
    parser.add_argument("--image_path", type=str, required=True, help="输入图像路径")
    parser.add_argument("--image_prompt", type=str, default="What is shown in this image? If there is a building, tell me the name of it and describe it in detail. Additionaly, tell me the weather in the image.", help="图像提示")
    parser.add_argument("--output_file", type=str, help="输出结果保存路径")
    parser.add_argument("--load_4bit", action="store_true", default=True, help="以4bit精度加载模型")
    parser.add_argument("--load_8bit", action="store_true", help="以8bit精度加载模型")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="最大生成token数")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载LLaVA模型
    print("正在加载微调后的LLaVA模型...")
    llava_model, llava_tokenizer, llava_processor, conv_mode = load_llava_official(
        args.llava_model_path, 
        args.llava_model_base,
        args.load_8bit,
        args.load_4bit, 
        args.device
    )
    
    # 处理图像
    print(f"正在分析图像: {args.image_path}")
    llava_output = process_with_llava_official(
        llava_model, 
        llava_tokenizer, 
        llava_processor, 
        conv_mode, 
        args.image_path, 
        args.image_prompt,
        args.max_new_tokens
    )
    print("\nLLaVA模型输出：")
    print(llava_output)
    
    results = {"llava_output": llava_output}
    
    # 如果指定了输出文件，保存结果
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {args.output_file}")

if __name__ == "__main__":
    main() 