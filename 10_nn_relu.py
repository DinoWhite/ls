import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5], [-1, 3]])
output = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10(
    './datasets', train=True, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Pony(nn.Module):
    def __init__(self):
        super(Pony, self).__init__()
        self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, x):
        output = self.sigmoid1(x)
        return output


writer = SummaryWriter("pony_sigmoid")
pony = Pony()
output = pony(input)

step = 0
for data in dataloader:
    imgs, targets = data
    output = pony(imgs)

    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)

    step = step + 1

writer.close()
