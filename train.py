import torch
from torch.utils.data import Dataset, DataLoader
from utils.dataloader import NoneDateSet
from torch import nn
from model.base_model import BaseModelResNet18
from matplotlib import pyplot as plt
from tqdm import tqdm
import config

def evaluate(model, train_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).to(torch.float32)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            # correct += torch.sum(preds == targets)

    epoch_loss = total_loss / len(train_loader.dataset)
    epoch_acc = correct.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    correct = 0

    for X, y in train_loader:
        data, targets = X.to(device), y.to(device)
        outputs = model(data)

        loss = criterion(outputs, targets)
        total_loss += loss.item() * data.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(train_loader.dataset), correct / len(train_loader.dataset)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_path = "../processed-data/train.npy"
    test_data_path = "../processed-data/test.npy"
    train_label_path = "../processed-data/train.csv"
    test_label_path = "../processed-data/test.csv"

    train_dataset = NoneDateSet(train_label_path, train_data_path)
    test_dataset = NoneDateSet(test_label_path, test_data_path)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    base_model = BaseModelResNet18(num_classes=4).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(params=base_model.parameters(), lr=1e-4, weight_decay=1e-5)
    num_epoch = config.NUM_EPOCH

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epoch*config.BATCH_SIZE)

    train_loss_list = []
    train_acc_list = []
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for epoch in tqdm(range(num_epoch)):
        epoch_loss, epoch_acc = train_epoch(base_model, train_loader, loss_fn, optimizer, device, scheduler)
        train_loss_list.append(epoch_loss)
        train_acc_list.append(epoch_acc)
        ax1.plot(train_loss_list, label='train loss')
        # ax2.plot(train_acc_list, label='train acc')

    fig.show()
