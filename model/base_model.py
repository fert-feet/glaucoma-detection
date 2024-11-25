from torchvision import models
from torch import nn
import torch

class BaseModelResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(BaseModelResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.pretrained_model_path = "./model/pre_trained_model/resnet/resnet18-5c106cde.pth"
        self.num_classes = num_classes
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            pretrained_model = torch.load(self.pretrained_model_path)
            self.model.load_state_dict(pretrained_model)

        num_feature = self.model.fc.in_features
        self.model.fc = nn.Linear(num_feature, self.num_classes)

    def forward(self, x):
        return self.model(x)

class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
