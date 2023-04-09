import torch
import numpy as np
import ModuleTrainer as MT
import PIL.Image as Image
import torch.utils.data as data

from models.SiamNet import SiamNet
from datasets.LEVIRCDDataset import LEVIRCDDataset, RunningMode


def train_siamnet(pt_path: str = None):
    if pt_path is None:
        ValueError(pt_path)

    trainer = MT.ModuleTrainer(dataset=LEVIRCDDataset(), module=SiamNet(), 
                               save_frequency=20, pt_path=pt_path, epoch=400, batch_size=1)
    trainer.train()

    torch.save(trainer.module, pt_path+"\\UNet_Finished.pt")