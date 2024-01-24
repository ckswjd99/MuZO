from shelf.models.resnet import ResNet34
from shelf.trainers import train, adjust_learning_rate, validate, adjust_learning_rate_warmup
from shelf.dataloaders import get_CIFAR10_dataset
from shelf.pruners import get_mask_L1, apply_mask, print_pruned_config, init_mask

import torch
import torch.nn as nn
import torch.optim as optim

import copy


# hyperparams
BATCH_SIZE = 128
LR = 1e-2
MOMENTUM = 0.95
WEIGHT_DECAY = 5e-4
PERTURB_EPS = 1e-5
ONE_WAY=True
SURVIVAL_RATE = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]
MASK_EPOCHS = [2, 2, 3, 3, 10, 10, 10, 10, 50, 50, 100]
TOTAL_EPOCHS = sum(MASK_EPOCHS)

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

masks = [
    get_mask_L1(model_temp, survival_ratio=survival_rate) for survival_rate in SURVIVAL_RATE
]

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

log = []

init_mask(model)

last_epoch = 0
for mask, mask_epoch in zip(masks, MASK_EPOCHS):
    apply_mask(model, mask)
    print_pruned_config(model)

    for epoch in range(last_epoch, last_epoch+mask_epoch):
        epoch_lr = adjust_learning_rate(optimizer, LR, epoch, epoch_freq=30/10, decay_rate=0.5 ** (1/10))
        if epoch - last_epoch < mask_epoch // 10:
            epoch_lr = epoch_lr * (epoch - last_epoch + 1) / (mask_epoch // 10)

        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch)

        print(f'Epoch {epoch+1}/{TOTAL_EPOCHS} lr: {epoch_lr:.4e} \t| train_acc: {train_acc * 100:.2f}%  \ttrain_loss: {train_loss:.4f}  \tval_acc: {val_acc * 100:.2f}%  \tval_loss: {val_loss:.4f}')

    last_epoch += mask_epoch