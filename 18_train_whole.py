import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *


train_data = torchvision.datasets.CIFAR10(
    './datasets', train=True, transform=torchvision.transforms.ToTensor(), download=True)

test_data = torchvision.datasets.CIFAR10(
    './datasets', train=False, transform=torchvision.transforms.ToTensor(), download=True)


# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print(train_data_size)
print(test_data_size)

# dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# network
pony = Pony()

# loss function
loss_fn = nn.CrossEntropyLoss()

# opimizer
learning_rate = 0.01
optimizer = optim.SGD(pony.parameters(), lr=learning_rate)

# writer
writer = SummaryWriter("whole")

# 训练网络的参数
total_train_step = 0
total_test_step = 0
epoch = 10  # 训练轮数

for i in range(epoch):
    print(f'{i+1} th round train')

    pony.train()  # 作用不大，可以不要
    for data in train_dataloader:
        imgs, targets = data
        outputs = pony(imgs)

        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step+1
        if total_train_step % 100 == 0:
            print(f"Training times {total_train_step}, Loss: {loss.item()}")
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试部分不用调节梯度
    pony.eval()  # 作用不大，可以不要
    total_test_loss = 0.
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = pony(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss+loss

            accuracy = (outputs.argmax(1) == targets).sum()
    print(f'whole test loss: {total_test_loss}')
    print(f'whole accuracy: {accuracy/test_data_size}')
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", accuracy/test_data_size,
                      total_test_step)
    total_test_step = total_test_step+1

    torch.save(pony, f'pony_{i}.pth')
    print('model saved')

writer.close()
