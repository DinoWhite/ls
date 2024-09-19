from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")

# writer.add_image()
# image_path = 'insect/train/bees_image/1092977343_cb42b38d62.jpg'
image_path = 'insect/train/ants_image/0013035.jpg'
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image("test", img_array, 2, dataformats='HWC')


# writer.add_scalar()
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)  # 标题 y轴 x轴

writer.close()
