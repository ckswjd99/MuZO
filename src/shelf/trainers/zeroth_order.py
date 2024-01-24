import torch
from tqdm import tqdm
import numpy as np


def train_zo_rge(train_loader, model, criterion, epoch, learning_rate=1e-7, weight_decay=5e-4, smoothing=1e-3, verbose=True, momentum=1.0, one_way=False, mask=None):
    model.eval()

    num_data = 0
    num_correct = 0
    sum_loss = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False) if verbose else train_loader
    max_grad = 0

    momentum_buffer = {}
    for name, param in model.named_parameters():
        momentum_buffer[name] = torch.zeros_like(param.data)

    for input, label in pbar:
        input = input.cuda()
        label = label.cuda()

        perturb_seed = np.random.randint(10000000)

        loss_original = criterion(model(input), label).item()

        _zo_perturb_rand(model, perturb_seed, smoothing = 1 * smoothing, mask=mask)
        output1 = model(input)
        loss_perturbed = criterion(output1, label).item()

        _zo_perturb_rand(model, perturb_seed, smoothing = -1 * smoothing, mask=mask)

        projected_gradient = (loss_perturbed - loss_original) / smoothing

        momentum_buffer = _zo_update(model, perturb_seed, projected_gradient, learning_rate, weight_decay, momentum_buffer, momentum, one_way=one_way, mask=mask)

        output = model(input)
        loss = criterion(output, label)

        _, predicted = torch.max(output.data, 1)
        num_data += label.size(0)
        num_correct += (predicted == label).sum().item()
        sum_loss += loss.item() * label.size(0)
    
        accuracy = num_correct / num_data
        avg_loss = sum_loss / num_data

        if verbose:
            pbar.set_postfix(train_accuracy=accuracy, train_loss=avg_loss)
        
    accuracy = num_correct / num_data
    avg_loss = sum_loss / num_data

    return accuracy, avg_loss

def _zo_perturb_rand(model, perturb_seed, smoothing=1.0, mask=None):
    torch.manual_seed(perturb_seed)
    
    for name, param in model.named_parameters():
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        if mask is not None:
            z = z * mask[name]
        param.data += z * smoothing

def _zo_update(model, perturb_seed, projected_gradient, learning_rate, weight_decay, momentum_buffer, momentum, one_way=False, mask=None):
    torch.manual_seed(perturb_seed)

    if one_way:
        projected_gradient = max(0, projected_gradient)

    for name, param in model.named_parameters():
        z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
        if mask is not None:
            z = z * mask[name]
        estimated_gradient = projected_gradient * z
        momentum_buffer[name] = momentum * momentum_buffer[name] + (1-momentum) * estimated_gradient

        if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
            param.data = param.data - learning_rate * (estimated_gradient + weight_decay * param.data)
        else:
            param.data = param.data - learning_rate * (estimated_gradient)
        
    return momentum_buffer

