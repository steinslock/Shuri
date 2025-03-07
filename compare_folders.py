import os

# 定义两个需要比较的文件夹路径
folder1 = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/finetuning_dataset/images_converted"
folder2 = "/home/qiangminc/codes/Shuri/relevance_eval/finetuning_new/finetuning_dataset/images"

# 获取两个文件夹中的所有文件名
files1 = set(os.listdir(folder1))
files2 = set(os.listdir(folder2))

# 找出只在第一个文件夹中存在的文件
only_in_folder1 = files1 - files2
# 找出只在第二个文件夹中存在的文件
only_in_folder2 = files2 - files1

# 输出结果
print(f"\n仅在 images_converted 文件夹中存在的文件 ({len(only_in_folder1)}):")
for file in sorted(only_in_folder1):
    print(f"- {file}")

print(f"\n仅在 images 文件夹中存在的文件 ({len(only_in_folder2)}):")
for file in sorted(only_in_folder2):
    print(f"- {file}")

# 计算总的不同文件数量
total_different = len(only_in_folder1) + len(only_in_folder2)
print(f"\n总共有 {total_different} 个不同的文件名")

# 输出两个文件夹中的文件总数，用于参考
print(f"\n文件夹统计:")
print(f"images_converted 文件夹: {len(files1)} 个文件")
print(f"images 文件夹: {len(files2)} 个文件") 