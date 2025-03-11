#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import csv
import re

# 定义路径
images_dir = '/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/finetuning_dataset/image/综合'
csv_file = '/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/finetuning_dataset/finetune_综合2.csv'
json_file = '/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/finetuning_dataset/finetune_综合.json'
output_json_file = '/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/finetuning_dataset/finetune_综合2.json'

def main():
    # 读取原始JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取所有图片文件并按数字顺序排序
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    # 提取数字部分并按数字大小排序
    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    image_files.sort(key=extract_number)
    
    # 读取CSV文件
    csv_data = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if len(row) >= 3:
                # 处理可能存在的转义字符
                human_prompt = row[1].replace('\\n', '\n')
                gpt_response = row[2].replace('\\n', '\n')
                csv_data.append((human_prompt, gpt_response))
    
    # 确保数据数量一致
    min_count = min(len(image_files), len(csv_data))
    print(f"处理 {min_count} 条数据")
    
    # 创建新的JSON数据
    new_data = []
    
    for i in range(min_count):
        image_name = image_files[i]
        image_id = os.path.splitext(image_name)[0]  # 去掉后缀获取ID
        
        # 创建新条目
        entry = {
            "id": image_id,
            "image": image_name,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{csv_data[i][0]}"
                },
                {
                    "from": "gpt",
                    "value": csv_data[i][1]
                }
            ]
        }
        new_data.append(entry)
    
    # 写入新的JSON文件
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    
    print(f"数据处理完成，已保存到 {output_json_file}")

if __name__ == "__main__":
    main() 