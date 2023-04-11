import os
import presets.UnetPresets as UP
import presets.SiameseUNetPresets as SP

if __name__ == '__main__':
    # UP.train_unet(os.path.abspath("computational_graphs/UNet"))
    # UP.evaluate_unet(os.path.abspath("computational_graphs/UNet/UNet_ep300_loss1.965.pt"), True, os.path.abspath("result"))
    # SP.train_SiameseUNet(os.path.abspath("computational_graphs/SiameseUNet"))
    SP.evaluate_SiameseUNet(os.path.abspath("computational_graphs/SiameseUNet/SiameseUNet_ep170_loss1.861.pt"), True, os.path.abspath("result/SiameseUNet"))