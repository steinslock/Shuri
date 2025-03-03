import os
import csv

def list_images_to_csv(folder_path, csv_file_path):
    # 获取文件夹中的所有文件名
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # 过滤出图片文件（假设图片文件扩展名为常见的图片格式）
    image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    
    # 将图片文件名写入CSV文件
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['image_name'])  # 写入CSV文件的表头
        for image_file in image_files:
            # 在图片名前添加路径前缀
            full_path = os.path.join(folder_path, image_file)
            csv_writer.writerow([full_path])

# 使用示例
folder_path = '/home/qiangminc/codes/Shuri_eval/relevance_eval/finetuning/微调用图片/正面'
csv_file_path = '/home/qiangminc/codes/Shuri_eval/relevance_eval/finetuning/微调用图片/name.csv'
list_images_to_csv(folder_path, csv_file_path)
