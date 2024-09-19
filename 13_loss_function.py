import torch
from torch import nn

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# l1
loss = nn.L1Loss()
result = loss(inputs, targets)
print(result)

# mse
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)
print(result_mse)

# cross entropy
x = torch.tensor([[0.1, 0.2, 0.3]])
y = torch.tensor([1])

x = torch.reshape(x, (1, 3))

loss_ce = nn.CrossEntropyLoss()
result_ce = loss_ce(x, y)
print(result_ce)
