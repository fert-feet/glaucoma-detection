import os
import shutil
import random

# 定义类别文件夹列表
category_folders = ["1_normal", "2_cataract", "2_glaucoma", "3_retina_disease"]
data_relative_path = "../../origin-data"
for category_folder in category_folders:
    # 获取当前类别文件夹下的所有图片文件列表
    category_folder_path = os.path.join(data_relative_path, category_folder)
    image_files = [f for f in os.listdir(category_folder_path) if f.endswith('.png')]
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