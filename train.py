import torch
import torch.nn as nn

from utils.dataloader import get_data_loaders
from utils.drawer import Plotter
from tqdm import tqdm
from model.base_model import BaseModelResNet18, SmallCNN
import matplotlib.pyplot as plt

# 定义数据路径
train_data_path = "../processed-data/pneumonia/train.npy"
test_data_path = "../processed-data/pneumonia/test.npy"
train_label_path = "../processed-data/pneumonia/train.csv"
test_label_path = "../processed-data/pneumonia/test.csv"

# 训练和评估
num_epochs = 50
train_loss_list, train_acc_list = [], []
test_loss_list, test_acc_list = [], []
batch_size = 32

# 获取数据加载器
train_loader, test_loader = get_data_loaders(train_data_path, train_label_path, test_data_path, test_label_path, batch_size)

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
base_model = SmallCNN(num_classes=2).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(base_model.parameters(), 1e-3)
# optimizer = torch.optim.Adam(base_model.parameters(), 1e-4, betas=(0.9, 0.999), weight_decay=0.0003)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# 训练函数
def epoch_train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    return train_loss, train_acc

# 评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    return test_loss, test_acc


for epoch in tqdm(range(num_epochs), desc='Epoch'):
    train_loss, train_acc = epoch_train(base_model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(base_model, test_loader, criterion, device)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    # scheduler.step(test_loss)


torch.save(base_model.state_dict(), f'./logs/model/pneumonia_mode_{base_model.get_name()}_acc_{test_acc_list[-1]}.pth')

draw_tool = Plotter(save_path='./logs/image')
draw_tool.plot_loss_and_accuracy(train_loss_list, test_loss_list, train_acc_list, test_acc_list)
