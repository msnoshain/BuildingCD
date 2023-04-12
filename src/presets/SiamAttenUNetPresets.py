import torch
import numpy as np
import PIL.Image as Image
import torch.utils.data as data
import tools.ModuleTrainer as MT
import tools.DataEvaluation as DE
import tools.reporters.QQEmailReporter as QQ

from datasets.Common import RunningMode
from models.SiamAttenUNet import SiamAttenUNet
from datasets.LEVIRCDDataset import LEVIRCDDataset


def train_SiamAttenUNet(out_path: str = None):
    if out_path is None:
        raise ValueError(out_path)

    trainer = MT.ModuleTrainer(dataset=LEVIRCDDataset(), module=SiamAttenUNet(),
                               save_frequency=10, pt_path=out_path, epoch=600, batch_size=4)

    trainer.report_loss = lambda loss, epoch: QQ.send_myself_QQEmail(
        "SiameseUNet Loss Report", "epoch: {}, loss: {}".format(epoch, loss))

    trainer.train()

    torch.save(trainer.module, out_path+"\\SiamAttenUNet_Finished.pt")