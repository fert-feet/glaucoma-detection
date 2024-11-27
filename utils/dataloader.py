import os

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from PIL import Image

class NoneDateSet(Dataset):
    def __init__(self, label_path, data_path):
        super(NoneDateSet, self).__init__()
        self.label_file = pd.read_csv(label_path)
        self.data = np.load(data_path)
        self.one_hot_labels = pd.get_dummies(self.label_file['class_name']).values # # one-hot encoding
        self.true_labels = torch.argmax(torch.from_numpy(self.one_hot_labels).to(torch.long), dim=1)
        self.length = self.label_file.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index], self.true_labels[index]

class EyesDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.labels = pd.read_csv(label_path)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.labels['class_name'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.tensor(self.data[idx], dtype=torch.float32)
        label = self.class_to_idx[self.labels.iloc[idx]['class_name']]
        return image, label

def get_data_loaders(train_data_path, train_label_path, test_data_path, test_label_path, batch_size=32):
    train_dataset = EyesDataset(train_data_path, train_label_path)
    test_dataset = EyesDataset(test_data_path, test_label_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.image_paths = []
        self.labels = []

        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for image_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, image_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label