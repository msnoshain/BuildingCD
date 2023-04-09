import os
from datasets.LEVIRCDDataset import LEVIRCDDataset
import presets.UnetPresets as UP
import presets.SiamNetPresets as SP

if __name__ == '__main__':
    # UP.train_unet(os.path.abspath("computational_graphs/UNet"))
    # UP.evaluate_unet(os.path.abspath("computational_graphs/UNet/UNet_ep300_loss1.965.pt"), True, os.path.abspath("result"))
    SP.train_siamnet(os.path.abspath("computational_graphs/SiamNet"))