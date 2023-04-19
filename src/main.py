import os
import presets.SiamAttenUNetPresets as SAP
import presets.SiamCBAMUNetPresets as SCP
import presets.UnetPresets as UP
import presets.SiameseUNetPresets as SP

if __name__ == '__main__':
    # UP.train_unet(os.path.abspath("computational_graphs/UNet"))
    # UP.evaluate_unet(os.path.abspath("computational_graphs/UNet/UNet_ep300_loss1.965.pt"), True, os.path.abspath("result"))
    # SP.train_SiameseUNet(os.path.abspath("computational_graphs/SiameseUNet"))
    # SP.evaluate_SiameseUNet(os.path.abspath("computational_graphs/SiameseUNet/SiameseUNet_ep170_loss1.861.pt"), True, os.path.abspath("result/SiameseUNet"))
    # SAP.train_SiamAttenUNet(os.path.abspath("computational_graphs/SiamAttenUNet"))
    # SAP.evaluate_SiamAttenUNet(os.path.abspath("computational_graphs/SiamAttenUNet/SiamAttenUNet_ep10_loss1.736.pt"), False, os.path.abspath("result/SiamAttenUNet"))
    # SAP.train_semi_finished_SiamAttenUNet(os.path.abspath("computational_graphs/SiamAttenUNet/SiamAttenUNet_ep200_loss2.213.pt"), os.path.abspath("computational_graphs/SiamAttenUNet/"))
    # SCP.train_SiamCBAMUNet(os.path.abspath("computational_graphs/SiamCBAMUNet"))
    SCP.train_semi_finished_SiamCBAMUNet(os.path.abspath("computational_graphs/SiamCBAMUNet/SiamAttenUNet_ep20_loss40.640.pt"), os.path.abspath("computational_graphs/SiamCBAMUNet"), 20)