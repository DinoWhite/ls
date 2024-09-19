import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, (-1, 1, 5, 5))

dataset = torchvision.datasets.CIFAR10(
    './datasets', train=True, transform=torchvision.transforms.ToTensor(), download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Pony(nn.Module):
    def __init__(self):
        super(Pony, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool1(x)

        return output


writer = SummaryWriter("pony_maxpool")
pony = Pony()
output = pony(input)

print(input)
print(output)
step = 0
for data in dataloader:
    imgs, targets = data
    output = pony(imgs)

    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)

    step = step + 1

writer.close()
