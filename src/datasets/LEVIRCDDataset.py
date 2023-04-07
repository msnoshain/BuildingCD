import os
import torch
import torch.utils.data as data
from torchvision.transforms import transforms
import PIL.Image as Image
from enum import Enum


class RunningMode(Enum):
    train = "train"
    test = "test"
    evaluation = "val"


class DataFetchingMode(Enum):
    raw = 0
    cat = 1


class LEVIRCDDataset(data.Dataset):
    running_mode: RunningMode
    data_fetching_mode: DataFetchingMode

    to_tenser_transformer = transforms.ToTensor()

    def __init__(self, running_mode: RunningMode = RunningMode.train, data_fetching_mode: DataFetchingMode = DataFetchingMode.cat):
        self.running_mode = running_mode
        self.data_fetching_mode = data_fetching_mode

    def __getitem__(self, index):
        running_mode_value = self.running_mode._value_

        img_x1 = Image.open(os.path.abspath("data/{}/A/{}_{}.png".format(running_mode_value, running_mode_value, index + 1))).resize([512, 512])
        img_x2 = Image.open(os.path.abspath("data/{}/B/{}_{}.png".format(running_mode_value, running_mode_value, index + 1))).resize([512, 512])
        img_y = Image.open(os.path.abspath("data/{}/label/{}_{}.png".format(running_mode_value, running_mode_value, index + 1))).resize([512, 512])


        if self.data_fetching_mode is DataFetchingMode.raw:
            return self.to_tenser_transformer(img_x1), self.to_tenser_transformer(img_x2), self.to_tenser_transformer(img_y)
        else:
            return torch.cat([self.to_tenser_transformer(img_x1), self.to_tenser_transformer(img_x2)], dim=0), self.to_tenser_transformer(img_y)

    def __len__(self):
        if self.running_mode is RunningMode.test:
            return 128
        elif self.running_mode is RunningMode.train:
            return 445
        else:
            return 64
