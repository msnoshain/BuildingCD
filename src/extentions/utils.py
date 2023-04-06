import torch

class utils():
    def is_using_cuda(device: torch.device):
        if device.type is "cpu": return False
        else: return True