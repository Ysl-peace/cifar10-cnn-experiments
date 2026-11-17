# -*- coding: utf-8 -*-
from pathlib import Path
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Model structure
class Net(nn.Module):
    def __init__(self):
        # 在构造函数中，实例化不同的layer组件，并赋给类成员变量
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 在前馈函数中，利用实例化的组件对网络进行搭建，并对输入Tensor进行操作，并返回Tensor类型的输出结果
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)#, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(net, loader, device, num_classes=10):
    """使用固定的模型参数，统计总体和各类别准确率（百分数）。"""
    net.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predicted = net(inputs).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            for label, prediction in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
                class_total[label] += 1
                class_correct[label] += int(label == prediction)
    if total == 0:
        raise ValueError('Cannot evaluate an empty dataset')
    class_accuracy = [
        100 * hits / count if count else None
        for hits, count in zip(class_correct, class_total)
    ]
    return 100 * correct / total, class_accuracy


if __name__ == '__main__':
    # 固定随机种子，让后续实验的随机过程更容易复现。
    torch.manual_seed(42)

    # 根据环境参数选取对应运行平台：cpu or gpu
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Running Device:', device)

    # 仅训练时随机增强；评估时保持图片不变。
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 利用torchvision提供的CIFAR10数据集类，实例化训练集和测试集提取类
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    # 同一批训练图片的评估版本，不使用随机翻转，也不会重复下载。
    train_eval_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=eval_transform)
    # 固定划分原始图片编号；验证图片及其增强版本都不参与训练。
    indices = torch.randperm(len(trainset), generator=torch.Generator().manual_seed(42)).tolist()
    train_indices, val_indices = indices[:45000], indices[45000:]
    valset = torch.utils.data.Subset(train_eval_set, val_indices)
    trainset = torch.utils.data.Subset(trainset, train_indices)
    train_eval_set = torch.utils.data.Subset(train_eval_set, train_indices)
    print(f'Train samples: {len(trainset)}, validation samples: {len(valset)}')

    # 利用torch提供的DataLoader, 实例化训练集DataLoader 和 测试集DataLoader
    Batch_size = 8
    trainLoader = torch.utils.data.DataLoader(trainset, batch_size=Batch_size, shuffle=True)
    train_eval_loader = torch.utils.data.DataLoader(train_eval_set, batch_size=Batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=Batch_size, shuffle=False)

    # CIFAR10 类别内容
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 卷积神经网络实例化
    net = Net().to(device)
    print(net)

    # 实例化损失函数和SGD优化子
    epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 卷积神经网络训练
    history = []
    training_seconds = 0.0
    for epoch in range(epochs):
        # evaluate() 会切换为评估模式，每轮训练前必须切换回来。
        net.train()
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        training_start = time.perf_counter()
        running_loss = 0.0
        epoch_loss_sum = 0.0
        epoch_samples = 0

        for i, data in enumerate(trainLoader, 0):
            inputs, labels = data
            # 将数据迁移到device中，如device为GPU，则将数据从CPU迁移到GPU；如device为CPU，则将数据从CPU迁移到CPU（即不作移动）
            inputs, labels = inputs.to(device), labels.to(device)

            # 清空参数的梯度值
            optimizer.zero_grad()

            # 前馈+反馈+优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计训练损失
            running_loss += loss.item()
            # loss 是当前批次的平均值，先乘样本数，再累计整轮总损失。
            epoch_loss_sum += loss.item() * labels.size(0)
            epoch_samples += labels.size(0)

            if i % 2000 == 1999:    # 每2000 mini-batches，打印一次
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        if device.startswith('cuda'):
            torch.cuda.synchronize()
        training_seconds += time.perf_counter() - training_start
        epoch_loss = epoch_loss_sum / epoch_samples
        val_accuracy, _ = evaluate(net, val_loader, device)
        history.append((epoch + 1, epoch_loss, val_accuracy))
        print(f'Epoch {epoch + 1}/{epochs}, mean training loss: {epoch_loss:.4f}, '
              f'validation accuracy: {val_accuracy:.2f} %')

    print(f'Training time: {training_seconds:.2f} seconds')
    output_dir = Path(__file__).resolve().parent
    with (output_dir / 'training_history.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'train_loss', 'val_accuracy'])
        writer.writerows(history)
    (output_dir / 'training_time.txt').write_text(
        f'Training time: {training_seconds:.2f} seconds\n', encoding='utf-8'
    )

    print('Finished Training')

    # 保存训练得到的权重和偏置；文件放在本脚本旁边。
    model_path = Path(__file__).resolve().parent / 'model.pt'
    torch.save(net.state_dict(), model_path)
    print('Model saved to:', model_path)

    # 开发阶段只评估训练集和验证集，官方测试集留待最终评估。
    train_accuracy, _ = evaluate(net, train_eval_loader, device)
    val_accuracy, class_accuracy = evaluate(net, val_loader, device)
    result_lines = [
        f'Train accuracy: {train_accuracy:.2f} %',
        f'Validation accuracy: {val_accuracy:.2f} %',
    ]
    for name, accuracy in zip(classes, class_accuracy):
        value = f'{accuracy:.2f} %' if accuracy is not None else 'N/A'
        result_lines.append(f'Validation accuracy of {name:>5s}: {value}')
    results = '\n'.join(result_lines)
    print(results)
    (output_dir / 'evaluation.txt').write_text(results + '\n', encoding='utf-8')
