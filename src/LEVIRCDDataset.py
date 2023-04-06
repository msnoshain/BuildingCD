import torch.utils.data as data
import PIL.Image as Image
from enum import Enum
import os


class Mode(Enum):
    train = "train"
    test = "test"
    valuation = "val"


class LEVIRCDDataset(data.Dataset):
    def __init__(self, x_transform=None, y_transform=None, mode=Mode.train):
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.mode = mode

    def __getitem__(self, index):
        mode_name=self.mode.name
        
        img_x1 = Image.open(os.path.abspath(os.path.join(
            os.getcwd(), "data/{}/A/{}_{}.png".format(mode_name, mode_name, index + 1))))
        img_x2 = Image.open(os.path.abspath(os.path.join(
            os.getcwd(), "data/{}/B/{}_{}.png".format(mode_name, mode_name, index + 1))))
        img_y = Image.open(os.path.abspath(os.path.join(
            os.getcwd(), "data/{}/label/{}_{}.png".format(mode_name, mode_name, index + 1))))

        if self.x_transform is not None:
            img_x1 = self.x_transform(img_x1)

        if self.y_transform is not None:
            img_x2 = self.y_transform(img_x2)

        return img_x1, img_x2, img_y

    def __len__(self):
        if self.mode is Mode.test: return 128
        if self.mode is Mode.train: return 445
        if self.mode is Mode.valuation: return 64
