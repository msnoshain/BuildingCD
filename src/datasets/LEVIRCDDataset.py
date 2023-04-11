import os
import torch
import PIL.Image as Image
import torch.utils.data as data

from torchvision.transforms import transforms
from datasets.Common import DataFetchingMode, RunningMode


class LEVIRCDDataset(data.Dataset):
    """
    LEVIR-CD数据集
    """

    to_tenser_transformer = transforms.ToTensor()

    def chunk_num(self): return self.split_num * self.split_num

    def __init__(self, running_mode: RunningMode = RunningMode.train, 
                 data_fetching_mode: DataFetchingMode = DataFetchingMode.concat,
                 split_num: int = 4):
        self.running_mode = running_mode
        self.data_fetching_mode = data_fetching_mode
        self.split_num = split_num

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

        if self.data_fetching_mode is DataFetchingMode.concat:
            return torch.cat([img_x1, img_x2], dim=0), img_y
        else:
            return img_x1, img_x2, img_y

    def __len__(self):
        if self.running_mode is RunningMode.test:
            return 128 * self.chunk_num()
        elif self.running_mode is RunningMode.train:
            return 445 * self.chunk_num()
        else:
            return 64 * self.chunk_num()
