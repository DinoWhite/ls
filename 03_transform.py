# from torchvision import transforms
# from PIL import Image

# img_path = "dataset/train/ants/0013035.jpg"
# img = Image.open(img_path)

# tensor_train = transforms.ToTensor()
# tensor_img = tensor_train(img)

# print(tensor_img)

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")

img = Image.open('izzy.jpg')
# print(img)

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)

writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1.5, 4.5, 2.5],
                                  [4.5, 8.5, 1.5])  # 三个值 分别表示三通道平均值，标准差
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])

writer.add_image("ToNorm", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize.size)
img_resize = trans_totensor(img_resize)

writer.add_image("Resize", img_resize)

# Compose
trans_resize_2 = transforms.Resize(128)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize 2", img_resize_2)

# RandomCrop
trans_random = transforms.RandomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
