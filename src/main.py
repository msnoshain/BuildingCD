import os
from models.unet import UNet
import presets.UnetPresets as UP
import torch.nn as nn

# UP.evaluate_unet(os.path.abspath("computational_graphs/u-net.pt"), False, os.path.abspath("result"))

m: nn.Module = UNet()
print(m.__class__.__name__)