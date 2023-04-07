from datasets.LEVIRCDDataset import LEVIRCDDataset
import DataTrainer as DT
import torch
import os
from models.unet import UNet

trainer = DT.DataTrainer(LEVIRCDDataset(), UNet(in_ch=6, out_ch=1))
trainer.train()
torch.save(trainer.module, os.path.abspath(os.path.join(os.getcwd(), "computational_graphs/u-net.pth")))