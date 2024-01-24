from shelf.models.resnet import ResNet34
from shelf.dataloaders import get_CIFAR10_dataset
from shelf.pruners import get_mask_L1, apply_mask, print_pruned_config, init_mask

import torch
import torch.nn as nn
import torch.optim as optim

# data
train_loader, val_loader = get_CIFAR10_dataset(root='./data', batch_size=128)

# model, criterion, optimizer
model = ResNet34(input_size=32, input_channel=3, num_output=10)
model = model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.95, weight_decay=5e-4)

init_mask(model)
mask30 = get_mask_L1(model, survival_ratio=0.3)
mask50 = get_mask_L1(model, survival_ratio=0.5)

# train one step
apply_mask(model, mask30)
inputs, targets = next(iter(train_loader))
inputs, targets = inputs.cuda(), targets.cuda()
outputs = model(inputs)
loss = criterion(outputs, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(list(model.named_parameters())[0])

# train another step
apply_mask(model, mask50)
inputs, targets = next(iter(train_loader))
inputs, targets = inputs.cuda(), targets.cuda()
outputs = model(inputs)
loss = criterion(outputs, targets)
optimizer.zero_grad()
loss.backward()
optimizer.step()
