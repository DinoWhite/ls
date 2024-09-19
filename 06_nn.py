import torch
import torch.nn as nn
import torch.nn.functional as F


class Pony(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input+1
        return output


pony = Pony()
x = torch.tensor(1.0)
output = pony(x)
print(output)
