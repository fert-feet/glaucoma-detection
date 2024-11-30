from torchvision import models
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
import torch

class BaseModelResNet18(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(BaseModelResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.pretrained_model_path = "./model/pre_trained_model/resnet/resnet18-5c106cde.pth"
        self.num_classes = num_classes
        pretrained_model = torch.load(self.pretrained_model_path)
        self.model.load_state_dict(pretrained_model)
        num_feature = self.model.fc.in_features
        self.model.fc = nn.Linear(num_feature, self.num_classes)
        self.name = "resnet18"

    def forward(self, x):
        return self.model(x)

    def get_name(self):
        return self.name

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()
        self.feas = []
        self.features_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 54 * 54, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
        self.name = "cnn"

    def forward(self, x):
        x = self.features_1(x)
        # self.feas.append(x)
        x = self.features_2(x)
        # self.feas.append(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_name(self):
        return self.name


# model = SmallCNN(num_classes=2).to('cuda')
# summary(model, torch.randn(1, 3, 224, 224).to('cuda'))
