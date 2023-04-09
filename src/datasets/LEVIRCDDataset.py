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

    split_num: int = 4

    def chunk_num(self): return self.split_num * self.split_num

    def __init__(self, running_mode: RunningMode = RunningMode.train):
        self.running_mode = running_mode

    def __getitem__(self, index):
        running_mode_value = self.running_mode._value_

        pic_index = int(index/self.chunk_num()) + 1
        chunk_index = index % self.chunk_num()
        i = int(chunk_index / self.split_num)
        j = int(chunk_index % self.split_num)
        a = 1024 / self.split_num

        box = j * a, i * a, (j + 1) * a, (i + 1) * a

        img_x1 = Image.open(os.path.abspath("data/{}/A/{}_{}.png".format(
            running_mode_value, running_mode_value, pic_index))).crop(box)
        img_x2 = Image.open(os.path.abspath("data/{}/B/{}_{}.png".format(
            running_mode_value, running_mode_value, pic_index))).crop(box)
        img_y = Image.open(os.path.abspath("data/{}/label/{}_{}.png".format(
            running_mode_value, running_mode_value, pic_index))).crop(box)

        img_x1 = self.to_tenser_transformer(img_x1)
        img_x2 = self.to_tenser_transformer(img_x2)
        img_y = self.to_tenser_transformer(img_y)

        return torch.cat([img_x1, img_x2], dim=0), img_y

    def __len__(self):
        if self.running_mode is RunningMode.test:
            return 128 * self.chunk_num()
        elif self.running_mode is RunningMode.train:
            return 445 * self.chunk_num()
        else:
            return 64 * self.chunk_num()
