import csv
import json

# 读取CSV文件并转换为JSON格式
def csv_to_json(csv_file_path, json_file_path):
    data = []
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append(row)
    
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

# 使用示例
csv_file_path = 'data.csv'
json_file_path = 'shurijo_data.json'
csv_to_json(csv_file_path, json_file_path)