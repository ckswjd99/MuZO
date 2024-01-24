from shelf.models.resnet import ResNet34
from shelf.trainers import train, adjust_learning_rate, validate, train_zo_rge
from shelf.dataloaders import get_CIFAR10_dataset
from shelf.pruners import get_mask_L1, apply_mask, print_pruned_config, init_mask

import torch
import torch.nn as nn
import torch.optim as optim

import copy


# hyperparams
EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-2
MOMENTUM = 0.95
WEIGHT_DECAY = 5e-4
PERTURB_EPS = 1e-5
ONE_WAY=True

NUM_CLASSES = 10


# data
train_loader, val_loader = get_CIFAR10_dataset(root='./data', batch_size=BATCH_SIZE)


# model, criterion, optimizer
model = ResNet34(input_size=32, input_channel=3, num_output=NUM_CLASSES)
model = model.cuda()
model.load_state_dict(torch.load('./saves/S0_init_acc9.74.pth'))

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

for epoch in range(EPOCHS):
    epoch_lr = adjust_learning_rate(optimizer, LR, epoch)
    
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    print(f'Epoch {epoch+1}/{EPOCHS}\tlr: {epoch_lr:.5f} \t| train_acc: {train_acc * 100:.2f}%  \ttrain_loss: {train_loss:.4f}  \tval_acc: {val_acc * 100:.2f}%  \tval_loss: {val_loss:.4f}')

