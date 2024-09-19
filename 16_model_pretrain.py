from torch import nn
import torchvision


vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)


train_data = torchvision.datasets.CIFAR10(
    './datasets', train=True, transform=torchvision.transforms.ToTensor(), download=True)


# 增加层级
print(vgg16_true)
vgg16_true.add_module('add_linear', nn.Linear(1000, 10))
# vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 修改层级
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)
