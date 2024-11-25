import os
import shutil
import random


def rename_jpeg_files(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)

    # 过滤出所有 JPEG 文件
    jpeg_files = [f for f in files if f.lower().endswith('.jpeg')]

    # 按照文件名排序（可选）
    jpeg_files.sort()

    # 遍历所有 JPEG 文件并重命名
    for index, old_name in enumerate(jpeg_files):
        # 构建新的文件名
        new_name = f"{index}_pneumonia.jpeg"

        # 获取旧文件的完整路径
        old_path = os.path.join(folder_path, old_name)

        # 获取新文件的完整路径
        new_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")

# 定义类别文件夹列表
category_folders = ["NORMAL", "PNEUMONIA"]
data_relative_path = "../../origin-data/pneumonia"

for category_folder in category_folders:
    # 获取当前类别文件夹下的所有图片文件列表
    category_folder_path = os.path.join(data_relative_path, category_folder)
    image_files = [f for f in os.listdir(category_folder_path) if f.endswith('.jpeg')]
    random.shuffle(image_files)  # 随机打乱图片文件顺序

    # 划分训练集和测试集
    num_images = len(image_files)
    num_train = int(num_images * 0.7)
    train_files = image_files[:num_train]
    test_files = image_files[num_train:]

    # 创建训练集和测试集对应的文件夹（如果不存在）
    train_folder = os.path.join(data_relative_path, "train")
    test_folder = os.path.join(data_relative_path, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 将图片文件复制到相应的训练集和测试集文件夹
    for file in train_files:
        shutil.copy(os.path.join(category_folder_path, file), os.path.join(train_folder, file))

    for file in test_files:
        shutil.copy(os.path.join(category_folder_path, file), os.path.join(test_folder, file))