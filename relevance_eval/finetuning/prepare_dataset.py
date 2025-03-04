import os
import json
import argparse
from PIL import Image
import shutil

def copy_image(image_path, target_path):
    """复制本地图像"""
    try:
        shutil.copy(image_path, target_path)
        return True
    except Exception as e:
        print(f"复制图像失败: {e}")
        return False

def create_dataset_structure(data_file, output_dir):
    """创建微调数据集的结构"""
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    entries = []
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for i, item in enumerate(data):
        image_file = f"image_{i}.jpg"
        image_path = os.path.join(images_dir, image_file)
        
        # 处理本地图像路径
        if "local_image_path" in item:
            try:
                success = copy_image(item["local_image_path"], image_path)
                if not success:
                    continue
            except Exception as e:
                print(f"复制图像失败: {e}")
                continue
        else:
            print(f"跳过条目 {i}: 没有本地图像路径")
            continue
        
        # 添加数据条目
        entries.append({
            "image_file": image_file,
            "prompt": item["prompt"],
            "response": item["response"]
        })
    
    # 写入metadata.jsonl文件
    with open(os.path.join(output_dir, "metadata.jsonl"), 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"数据集准备完成，共 {len(entries)} 条记录")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="准备LLaVA微调数据集")
    parser.add_argument("--data_file", type=str, required=True, help="输入数据JSON文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出数据集目录")
    
    args = parser.parse_args()
    create_dataset_structure(args.data_file, args.output_dir) 