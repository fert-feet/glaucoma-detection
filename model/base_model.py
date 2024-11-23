from torchvision import models
from torch import nn
import torch

class BaseModel(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(BaseModel, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.pretrained_model_path = "./model/pre_trained_model/resnet/resnet18-5c106cde.pth"
        self.num_classes = num_classes
        pretrained_model = torch.load(self.pretrained_model_path)
        self.model.load_state_dict(pretrained_model)
        num_feature = self.model.fc.in_features
        self.model.fc = nn.Linear(num_feature, self.num_classes)

    def forward(self, x):
        return self.model(x)
