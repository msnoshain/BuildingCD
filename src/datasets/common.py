from enum import Enum


class RunningMode(Enum):
    """
    程序运行模式：训练、测试或评估
    """

    train = "train"
    test = "test"
    evaluation = "val"

class DataFetchingMode(Enum):
    concat = 0
    raw = 1