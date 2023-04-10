from email.header import Header
import torch
import smtplib
from email.mime.text import MIMEText
import ModuleTrainer as MT
from models.SiameseUNet import SiameseUNet
from datasets.LEVIRCDDataset import LEVIRCDDataset
from reporters.QQEmailReporter import send_myself_QQEmail


def send_loss_to_email(loss: float, epoch: int):
    send_myself_QQEmail("SiameseUNet Loss Report", "epoch: {}, loss: {}".format(epoch, loss))


def train_SiameseUNet(pt_path: str = None):
    if pt_path is None:
        raise ValueError(pt_path)

    trainer = MT.ModuleTrainer(dataset=LEVIRCDDataset(), module=SiameseUNet(),
                               save_frequency=10, pt_path=pt_path, epoch=600, batch_size=6)

    trainer.report_loss = send_loss_to_email

    trainer.train()

    torch.save(trainer.module, pt_path+"\\SiameseUNet_Finished.pt")
