import torch
import torch.utils.data as data
from torchvision.transforms import transforms
import PIL.Image as Image
from enum import Enum
import os


class RunningMode(Enum):
    train = "train"
    test = "test"
    valuation = "val"


class DataFetchingMode(Enum):
    raw = 0
    cat = 1


class LEVIRCDDataset(data.Dataset):
    running_mode: RunningMode = RunningMode.train
    data_fetching_mode: DataFetchingMode = DataFetchingMode.cat

    to_tenser_transformer = transforms.ToTensor()

    def __init__(self):
        pass

    def __getitem__(self, index):
        running_mode_name = self.running_mode.name

        img_x1 = Image.open(os.path.abspath(os.path.join(
            os.getcwd(), "data/{}/A/{}_{}.png".format(running_mode_name, running_mode_name, index + 1))))
        img_x2 = Image.open(os.path.abspath(os.path.join(
            os.getcwd(), "data/{}/B/{}_{}.png".format(running_mode_name, running_mode_name, index + 1))))
        img_y = Image.open(os.path.abspath(os.path.join(
            os.getcwd(), "data/{}/label/{}_{}.png".format(running_mode_name, running_mode_name, index + 1))))

        if self.data_fetching_mode is DataFetchingMode.raw:
            return self.to_tenser_transformer(img_x1), self.to_tenser_transformer(img_x2), self.to_tenser_transformer(img_y)
        else:
            return torch.cat([self.to_tenser_transformer(img_x1), self.to_tenser_transformer(img_x2)], dim=1)

    def __len__(self):
        if self.running_mode is RunningMode.test:
            return 128
        if self.running_mode is RunningMode.train:
            return 445
        if self.running_mode is RunningMode.valuation:
            return 64
