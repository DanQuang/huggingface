import torch

def count_trainable_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)