import torch
from tools.reporters.QQEmailReporter import send_myself_QQEmail
import tools.ModuleTrainer as MT
from models.SiameseUNet import SiameseUNet
from datasets.LEVIRCDDataset import LEVIRCDDataset


def train_SiameseUNet(out_path: str = None):
    if out_path is None:
        raise ValueError(out_path)

    trainer = MT.ModuleTrainer(dataset=LEVIRCDDataset(), module=SiameseUNet(),
                               save_frequency=10, pt_path=out_path, epoch=600, batch_size=6)

    trainer.report_loss = lambda loss, epoch: send_myself_QQEmail(
        "SiameseUNet Loss Report", "epoch: {}, loss: {}".format(epoch, loss))

    trainer.train()

    torch.save(trainer.module, out_path+"\\SiameseUNet_Finished.pt")


def train_semi_finished_SiameseUNet(pt_path: str = None, out_path: str = None):
    if pt_path is None:
        raise ValueError(pt_path)

    if out_path is None:
        raise ValueError(out_path)

    module = torch.load(pt_path, map_location=torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"))

    trainer = MT.ModuleTrainer(dataset=LEVIRCDDataset(), module=module,
                               save_frequency=10, pt_path=out_path, epoch=600, batch_size=6)

    trainer.report_loss = lambda loss, epoch: send_myself_QQEmail(
        "SiameseUNet Loss Report", "epoch: {}, loss: {}".format(epoch, loss))

    trainer.train()

    torch.save(trainer.module, out_path+"\\SiameseUNet_Finished.pt")
