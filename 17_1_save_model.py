import torch
import torchvision
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# mothod 1 ( moedl + param )
torch.save(vgg16, "vgg16_method1.pth")

# method 2 (param)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 陷阱
class Pony(nn.Module):
    def __init__(self):
        super(Pony, self).__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool1(x)

        return output


pony = Pony()
torch.save(pony, 'pony_method1.pth')
