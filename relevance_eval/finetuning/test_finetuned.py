import torch
from PIL import Image
from transformers import AutoProcessor
from peft import PeftModel, PeftConfig
from transformers import LlavaForConditionalGeneration

# 加载微调后的模型和处理器
model_path = "./llava_lora_finetuned"
base_model_id = "llava-hf/llava-1.5-7b-hf"

# 加载配置和基础模型
config = PeftConfig.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(
    base_model_id, 
    torch_dtype=torch.float16,
    device_map="auto"
)

# 加载LoRA权重
model = PeftModel.from_pretrained(model, model_path)
processor = AutoProcessor.from_pretrained(model_path)

# 加载测试图像
image_path = "/path/to/test_shurijo.jpg"
image = Image.open(image_path).convert("RGB")

# 准备提示
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "描述这张图片中的建筑物，并提供详细信息。"},
        ],
    },
]

prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(image, text=prompt, return_tensors="pt").to(model.device, torch.float16)

# 生成回复
generated_ids = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

# 解码并打印结果
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
assistant_response = generated_text.split("ASSISTANT:")[-1].strip()
print("微调后的模型输出：")
print(assistant_response) 