import torch
import torchvision
from torch import nn

# method 1
vgg16 = torch.load("vgg16_method1.pth")
print(vgg16)

# method 1
vgg16 = torchvision.models.vgg16(pretrained=False)
param = torch.load("vgg16_method2.pth")
vgg16.load_state_dict(param)
print(vgg16)

# 陷阱
class Pony(nn.Module):
    def __init__(self):
        super(Pony, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool1(x)

        return output


pony = torch.load("pony_method1.pth")
print(pony)
