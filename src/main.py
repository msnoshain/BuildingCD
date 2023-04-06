from datasets.LEVIRCDDataset import LEVIRCDDataset, RunningMode
import PIL.Image as Image
import torch

img1, img2=LEVIRCDDataset(mode=RunningMode.train).__getitem__(0)
img1.show()