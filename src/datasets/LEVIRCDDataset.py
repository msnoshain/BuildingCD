import os
import torch
import PIL.Image as Image
import torch.utils.data as data

from enum import Enum
from torchvision.transforms import transforms


class RunningMode(Enum):
    """
    程序运行模式：训练、测试或评估
    """

    train = "train"
    test = "test"
    evaluation = "val"


class LEVIRCDDataset(data.Dataset):
    """
    LEVIR-CD数据集
    """

    running_mode: RunningMode

    to_tenser_transformer = transforms.ToTensor()

    def __init__(self, running_mode: RunningMode = RunningMode.train):
        self.running_mode = running_mode

    def __getitem__(self, index):
        running_mode_value = self.running_mode._value_

        img_x1 = Image.open(os.path.abspath("data/{}/A/{}_{}.png".format(
            running_mode_value, running_mode_value, index + 1))).resize([512, 512])
        img_x2 = Image.open(os.path.abspath("data/{}/B/{}_{}.png".format(
            running_mode_value, running_mode_value, index + 1))).resize([512, 512])
        img_y = Image.open(os.path.abspath("data/{}/label/{}_{}.png".format(
            running_mode_value, running_mode_value, index + 1))).resize([512, 512])

        img_x1 = self.to_tenser_transformer(img_x1)
        img_x2 = self.to_tenser_transformer(img_x2)
        img_y = self.to_tenser_transformer(img_y)

        return torch.cat([img_x1, img_x2], dim=0), img_y

    def __len__(self):
        if self.running_mode is RunningMode.test:
            return 128
        elif self.running_mode is RunningMode.train:
            return 445
        else:
            return 64
