import argparse
import os
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path


def merge_lora(args):
    model_path = args.model_path
    
    # 如果指定了checkpoint，构建完整的checkpoint路径
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(model_path, f"checkpoint-{args.checkpoint}")
        if os.path.exists(checkpoint_path):
            model_path = checkpoint_path
        else:
            raise ValueError(f"指定的checkpoint不存在: {checkpoint_path}")
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="LoRA模型的基础路径")
    parser.add_argument("--model-base", type=str, required=True, help="基础模型路径")
    parser.add_argument("--save-model-path", type=str, required=True, help="保存合并后模型的路径")
    parser.add_argument("--checkpoint", type=str, default=None, help="指定要合并的checkpoint编号，例如'1000'表示使用checkpoint-1000")

    args = parser.parse_args()

    merge_lora(args)
