import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    './datasets', train=True, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Pony(nn.Module):
    def __init__(self):
        super(Pony, self).__init__()
        self.linear1 = nn.Linear(196608, 10)

    def forward(self, x):
        output = self.linear1(x)
        return output


pony = Pony()

for data in dataloader:
    imgs, targets = data

    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)

    output = pony(output)
    print(output.shape)
