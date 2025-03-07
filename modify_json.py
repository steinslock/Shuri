from PIL import Image
import os

image_folder = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/finetuning_dataset/images_converted"

for img_file in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_file)
    try:
        with Image.open(img_path) as img:
            img.verify()  # 只检查，不加载
        print(f"Checked: {img_file} ✅")
    except Exception as e:
        print(f"Corrupted image: {img_file}, Error: {e}")