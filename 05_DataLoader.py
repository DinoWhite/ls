import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(
    './datasets', train=False, transform=torchvision.transforms.ToTensor())


test_loader = DataLoader(dataset=test_data, batch_size=4,
                         shuffle=True, num_workers=0, drop_last=False)

# First picture
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")


for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images(f"Epoch: {epoch}", imgs, step)
        step = step+1

writer.close()
