import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from model.base_model import SmallCNN


class Plotter:
    def __init__(self, save_path=None, fig_size=(10, 8)):
        self.save_path = save_path
        self.fig_size = fig_size

    def plot_loss_and_accuracy(self, train_loss_list, test_loss_list, train_acc_list, test_acc_list):

        plt.figure(figsize=self.fig_size)

        # 绘制损失曲线
        plt.subplot(2, 1, 1)
        plt.plot(train_loss_list, label='Train Loss')
        plt.plot(test_loss_list, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # 绘制准确率曲线
        plt.subplot(2, 1, 2)
        plt.plot(train_acc_list, label='Train Acc')
        plt.plot(test_acc_list, label='Test Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        if self.save_path:
            plt.savefig(os.path.join(self.save_path, "loss_and_accuracy.png"))
            print(f"saved loss and accuracy figure to {self.save_path}")

        plt.show()

    def show_train_img(self, npy_file, csv_file):
        datas = np.load(npy_file)
        labels = pd.read_csv(csv_file)['class_name'].values
        class_name_list = [f"{labels[i]}" for i in range(len(labels))]

        random_num = random_int(0, len(labels) - 2)
        image_np_list = datas[random_num:random_num + 2]
        image_title_list = class_name_list[random_num:random_num + 2]
        image_np_list = np.transpose(image_np_list, (0, 2, 3, 1))

        fig, axs = plt.subplots(1, 2)
        for i, ax in enumerate(axs):
            ax.axis('off')
            ax.imshow(image_np_list[i])
            ax.set_title(image_title_list[i])
        fig.show()
        fig.savefig(os.path.join(self.save_path, "train_input_img.png"))


    def show_origin_img(self, origin_file):
        class_name_list = ["NORMAL", "PNEUMONIA"]
        normal_files = os.listdir(os.path.join(origin_file, class_name_list[0]))
        abnormal_files = os.listdir(os.path.join(origin_file, class_name_list[1]))
        normal_img = normal_files[0]
        abnormal_img = abnormal_files[0]
        img_list = [normal_img, abnormal_img]

        fig, axs = plt.subplots(1, 2)
        for i, ax in enumerate(axs):
            img = Image.open(os.path.join(origin_file, class_name_list[i], img_list[i])).convert('RGB')
            ax.axis('off')
            ax.imshow(img)
            ax.set_title(class_name_list[i])
        fig.show()
        fig.savefig(os.path.join(self.save_path, "origin_img.png"))


    def show_aug_train_img(self, origin_file_path, row=2, columns=4, img_num=8):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转，概率为0.5
            transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转，概率为0.5
            transforms.RandomRotation(degrees=15),  # 随机调整亮度、对比度、饱和度和色调
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        file_name_list = os.listdir(origin_file_path)
        label_list = [file_name.split('.')[0].split('_')[1] for file_name in file_name_list][:img_num]
        full_path_list = [os.path.join(origin_file_path, file_name) for file_name in file_name_list][:img_num]
        img_list = [transform(Image.open(full_path).convert('RGB')) for full_path in full_path_list]
        img_list = [np.transpose(img, (1, 2, 0)) for img in img_list]

        fig, axs = plt.subplots(row, columns)
        for i in range(row):
            for j in range(columns):
                axs[i, j].axis('off')
                axs[i, j].imshow(img_list[i * columns + j], cmap='viridis')
                axs[i, j].set_title(label_list[i * columns + j])
        fig.show()
        fig.savefig(os.path.join(self.save_path, "aug_train_img.png"))

    def show_layer_output_img(self, pth_file):
        model = SmallCNN(num_classes=2).to('cuda')
        model.load_state_dict(torch.load(pth_file))

        target_layer = model.features[0]
        hook_handle = target_layer.register_forward_hook(hook_fn)
        data = np.load("../processed-data/pneumonia/test.npy")[0]
        data = torch.from_numpy(np.expand_dims(data, axis=0)).to('cuda')
        output = model(data)
        target_layer_output = hook_fn.target_layer_output
        print(target_layer_output.shape)
        hook_handle.remove()

def random_int(start, end):
    return np.random.randint(start, end)

# 定义一个 hook 函数
def hook_fn(module, input, output):
    # 保存第一个卷积层的输出
    hook_fn.first_conv_output = output


