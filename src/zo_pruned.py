import sys
sys.path.append('../../')

import argparse

from shelf.models.resnet import ResNet34
from shelf.trainers import train, adjust_learning_rate, validate, train_zo_rge
from shelf.dataloaders import get_CIFAR10_dataset
from shelf.pruners import get_mask_L1, apply_mask, init_mask, print_pruned_config

import torch
import torch.nn as nn
import torch.optim as optim

import copy


# hyperparams
EPOCHS = 25
BATCH_SIZE = 128
LR = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
PERTURB_EPS = 5e-3
ONE_WAY=False
SURVIVAL_RATE = 0.005

NUM_CLASSES = 10


# data
train_loader, val_loader = get_CIFAR10_dataset(root='./data', batch_size=BATCH_SIZE)


# model, criterion
model = ResNet34(input_size=32, input_channel=3, num_output=NUM_CLASSES)
model = model.cuda()
model_temp = copy.deepcopy(model)

fo_init = torch.load('./saves/S0_init_acc9.74.pth')
fo_best = torch.load('./saves/S0_best_acc88.92.pth')

model.load_state_dict(fo_init)
model_temp.load_state_dict(fo_best)

mask = get_mask_L1(model_temp, SURVIVAL_RATE)
init_mask(model)
apply_mask(model, mask)
print_pruned_config(model)

criterion = nn.CrossEntropyLoss()

log = []

# train with perturbation
for epoch in range(EPOCHS):
    train_acc, train_loss = train_zo_rge(
        train_loader, model, criterion, epoch,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        smoothing=PERTURB_EPS,
        momentum=MOMENTUM,
        one_way=ONE_WAY,
        mask=mask
    )
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    epoch_log = {
        'epoch': epoch,
        'learning_rate': LR,
        'weight_decay': WEIGHT_DECAY,
        'epsilon': PERTURB_EPS,
        'momentum': MOMENTUM,
        'one_way': ONE_WAY,

        'survival_rate': SURVIVAL_RATE,

        'train_acc': train_acc,
        'train_loss': train_loss,
        'val_acc': val_acc,
        'val_loss': val_loss
    }

    print(f'Epoch {epoch+1}/{EPOCHS} \t| train_acc: {train_acc * 100:.2f}%  \ttrain_loss: {train_loss:.4f}  \tval_acc: {val_acc * 100:.2f}%  \tval_loss: {val_loss:.4f}')

    log.append(epoch_log)

torch.save({
        'model': model.state_dict(),
        'epoch_start': 0,
        'epoch_end': EPOCHS,
        'final_accuracy': val_acc,
        'final_loss': val_loss,
        'log': log
    }, 
    f'./saves/mutate_zo/S0_best_acc{val_acc * 100:.2f}.pth'
)