import os
import presets.UnetPresets as UP

UP.train_unet(os.path.abspath("computational_graphs/UNet"))