import os
import presets.UnetPresets as UP

UP.evaluate_unet(os.path.abspath("computational_graphs/u-net.pt"), False, os.path.abspath("result"))