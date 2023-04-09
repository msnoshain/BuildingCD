import os

import torch
from datasets.LEVIRCDDataset import LEVIRCDDataset
from models.SiamNet import SiamNet
import presets.UnetPresets as UP
import presets.SiamNetPresets as SP
import ModuleTrainer as MT

if __name__ == '__main__':
    # UP.train_unet(os.path.abspath("computational_graphs/UNet"))
    # UP.evaluate_unet(os.path.abspath("computational_graphs/UNet/UNet_ep300_loss1.965.pt"), True, os.path.abspath("result"))
    # SP.train_siamnet(os.path.abspath("computational_graphs/SiamNet"))
    trainer = MT.ModuleTrainer(dataset=LEVIRCDDataset(), module=torch.load(os.path.abspath("computational_graphs/SiamNet/SiamNet_ep120_loss48.941.pt"), map_location=torch.device("cuda")),
                               save_frequency=20, pt_path=os.path.abspath("computational_graphs/SiamNet"), epoch=600, batch_size=2)
    trainer.train()

    torch.save(trainer.module, os.path.abspath(
        "computational_graphs/SiamNet")+"\\UNet_Finished.pt")

