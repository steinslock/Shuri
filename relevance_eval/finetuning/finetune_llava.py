import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    TaskType
)
import bitsandbytes as bnb

# 1. 配置基本参数
OUTPUT_DIR = "./llava_lora_finetuned"
LORA_R = 16                # LoRA秩
LORA_ALPHA = 32            # LoRA缩放因子
LORA_DROPOUT = 0.05        # LoRA dropout率
LEARNING_RATE = 5e-5       # 学习率
BATCH_SIZE = 2             # 批次大小
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
NUM_EPOCHS = 5             # 训练轮数
MAX_SEQ_LENGTH = 512       # 最大序列长度
DATASET_PATH = "./shurijo_dataset"  # 数据集路径

# 2. 定义自定义数据集类
class LlavaFinetuningDataset(Dataset):
    def __init__(self, dataset_path, processor, max_length):
        self.processor = processor
        self.max_length = max_length
        self.data = []
        
        # 加载数据集
        with open(os.path.join(dataset_path, "metadata.jsonl"), "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    "image_path": os.path.join(dataset_path, "images", item["image_file"]),
                    "prompt": item["prompt"],
                    "response": item["response"]
                })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        
        # 准备对话格式
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": item["response"],
            }
        ]
        
        # 处理输入
        prompt = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=prompt,
            images=image,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 处理标签（将输入ID复制为标签）
        labels = inputs["input_ids"].clone()
        
        # 将非助手回复部分的标签设为-100（在计算损失时忽略）
        assistant_start = prompt.find("ASSISTANT:")
        if assistant_start != -1:
            assistant_token_idx = len(self.processor.tokenizer(prompt[:assistant_start], truncation=True)["input_ids"])
            labels[:, :assistant_token_idx] = -100
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels.squeeze(),
            "pixel_values": inputs.pixel_values.squeeze()
        }

# 3. 数据整理函数
def collate_fn(examples):
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_mask = torch.stack([example["attention_mask"] for example in examples])
    labels = torch.stack([example["labels"] for example in examples])
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values
    }

# 4. 主函数
def main():
    # 加载模型和处理器
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # 使用4位量化加载模型
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb.nn.modules.Linear4bit.quantize_config(),
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # 为4位量化训练准备模型
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
    )
    
    # 应用LoRA适配器到模型
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 显示可训练参数占比
    
    # 准备数据集
    train_dataset = LlavaFinetuningDataset(DATASET_PATH, processor, MAX_SEQ_LENGTH)
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        load_best_model_at_end=True,
        label_names=["labels"],
    )
    
    # 配置Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print(f"微调完成，模型已保存到 {OUTPUT_DIR}")

# 5. 程序入口
if __name__ == "__main__":
    main() 