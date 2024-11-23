from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class EyeDateSet(Dataset):
    def __init__(self, label_path, data_path):
        super(EyeDateSet,self).__init__()
        self.label_file = pd.read_csv(label_path)
        self.data = np.load(data_path)
        self.labels = pd.get_dummies(self.label_file['class_name']).values
        self.length = self.label_file.shape[0] # one-hot encoding

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index], self.labels[index]