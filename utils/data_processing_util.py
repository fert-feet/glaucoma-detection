import os
import numpy as np
from PIL import Image
import glob
from torchvision import transforms
from tqdm import tqdm
import pandas as pd

def processing(folder_path):
    processed_data_list = []
    file_name_label_list = []

    files = glob.glob(os.path.join(folder_path, '*.jpeg'))
    for file in tqdm(files, desc='Processing'):
        file_name_label_list.append(extract_file_information(file))
        processed_data = data_processing(file)
        processed_data_list.append(processed_data)

    file_category_pd = pd.DataFrame(list(file_name_label_list))
    file_category_pd.columns = ["file_name", "class_name"]

    processed_data_np = np.array(processed_data_list)

    return processed_data_np, file_category_pd

def data_processing(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(1), # 单通道
        transforms.Resize((224, 224)),  # ResNet18通常要求输入图像大小为224x224
        transforms.ToTensor(),
        transforms.Normalize([0.485, ], [0.229, ])
        ])

    image = Image.open(image_path)
    image = transform(image)
    return image

def extract_file_information(file_path):
    file_name = file_path.split('/')[-1].split('.')[0]
    file_label = file_name.split('_')[1]

    return list((file_name, file_label))

if __name__ == '__main__':
    train_folder_path = "../../origin-data/pneumonia/train"
    test_folder_path = "../../origin-data/pneumonia/test"

    processed_data_path = "../../processed-data/pneumonia"
    category_list = ["normal", "cataract", "glaucoma", "retina_disease"]

    train_data, train_label = processing(train_folder_path)
    test_data, test_label = processing(test_folder_path)

    np.save(os.path.join(processed_data_path, "train"), train_data)
    np.save(os.path.join(processed_data_path, "test"), test_data)

    train_label.to_csv(os.path.join(processed_data_path, "train.csv"))
    test_label.to_csv(os.path.join(processed_data_path, "test.csv"))


