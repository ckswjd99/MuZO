import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import copy

def init_mask(model):
    for name, module in model.named_modules():
        if 'conv' in name or 'fc' in name:
            prune.identity(module, name='weight')

def get_mask_L1(model, survival_ratio):
    model_temp = copy.deepcopy(model)

    for name, module in model_temp.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=(1-survival_ratio))

    mask = {}
    for name, module in model_temp.named_buffers():
        if '_mask' in name:
            mask[name] = module

    return mask

def apply_mask(model, mask):
    for name, buffer in model.named_buffers():
        if '_mask' in name:
            buffer.copy_(mask[name])

def print_pruned_config(model):
    num_elements = 0
    num_alive_elements = 0

    for name, param in model.named_parameters():
        num_elements += param.numel()

    for name, buffer in model.named_buffers():
        if '_mask' in name:
            num_alive_elements += buffer.sum().item()
    
    print(f'Params {num_alive_elements/num_elements * 100:.2f}% alive ({int(num_alive_elements)}/{int(num_elements)}) parameters')