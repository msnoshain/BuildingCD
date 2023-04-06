from LEVIRCDDataset import LEVIRCDDataset, Mode
import PIL.Image as Image
import torch

img1, img2=LEVIRCDDataset(mode=Mode.train).__getitem__(0)
img1.show()