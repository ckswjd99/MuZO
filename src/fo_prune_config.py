from shelf.models.resnet import ResNet34
from shelf.trainers import train, adjust_learning_rate, validate, train_zo_rge
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
# SURVIVAL_RATE = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1]
# EPOCHS = [50, 50, 50, 50, 100, 100, 100, 200, 200, 200, 200]
SURVIVAL_RATE = [0.002]
EPOCHS = [50]

NUM_CLASSES = 10


# data
train_loader, val_loader = get_CIFAR10_dataset(root='./data', batch_size=BATCH_SIZE)


for survival_rate, num_epoch in zip(SURVIVAL_RATE, EPOCHS):

    # model, criterion
    model = ResNet34(input_size=32, input_channel=3, num_output=NUM_CLASSES)
    model = model.cuda()
    model_temp = copy.deepcopy(model)

    fo_init = torch.load('./saves/S0_init_acc9.74.pth')
    fo_best = torch.load('./saves/S0_best_acc88.92.pth')

    model.load_state_dict(fo_init)
    model_temp.load_state_dict(fo_best)

    mask = get_mask_L1(model_temp, survival_ratio=survival_rate)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    init_mask(model)
    apply_mask(model, mask)
    print_pruned_config(model)


    for epoch in range(num_epoch):
        epoch_lr = adjust_learning_rate(optimizer, LR, epoch, epoch_freq=30 * num_epoch / 200 / 10, decay_rate=0.5 ** (0.1))
        
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)
        val_acc, val_loss = validate(val_loader, model, criterion, epoch)

        print(f'Epoch {epoch+1}/{num_epoch}\tlr: {epoch_lr:6f} | train_acc: {train_acc * 100:.2f}%  \ttrain_loss: {train_loss:.4f}  \tval_acc: {val_acc * 100:.2f}%  \tval_loss: {val_loss:.4f}')
