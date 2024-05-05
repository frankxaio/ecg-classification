import dataclasses
from enum import Enum
from typing import Dict, Union

from ecg_tools.augmentations import Compose, RandomNoise, RandomShift


class Mode(Enum):
    train = "train"
    eval = "eval"


@dataclasses.dataclass()
class DatasetConfig:
    batch_size: int = 512 # 原本是 64，調小 bath size，使用 bz=16 訓練，搭配 lr=0.00005，使用 bz = 256 評估訓練完的模型
    num_workers: int = 0  # 多線程設定，設成0 itertools.chain 才不會出錯，但是會很慢
    path: Dict = dataclasses.field(default_factory=lambda: {
        # Mode.train: "..\\data\\mitbih_ptbdb_train.csv",
        # Mode.eval: "..\\data\\mitbih_ptbdb_test.csv"

        # absolute path
        Mode.train: "C:\\Users\\Xaio\\Documents\\GitHub\\ecg-classification\\data\\mitbih_ptbdb_train.csv",
        Mode.eval: "C:\\Users\\Xaio\\Documents\\GitHub\\ecg-classification\\data\\mitbih_ptbdb_test.csv"

    })
    transforms: Dict = dataclasses.field(default_factory=lambda: {
        Mode.train: Compose([RandomNoise(0.05, 0.5),
                             RandomShift(10, 0.5)]),
        Mode.eval: lambda x: x})


@dataclasses.dataclass()
class ModelConfig:
    num_layers: int = 1       # Transformer 的層數，也就是 Encoder 的 N，原本 N=6。代表重複執行的次數，且每次重複執行的參數都不一樣
                              # 為了減少晶片的面積與實作難度，將層數減少到 4 層
    signal_length: int = 187  # 原本是 187
    num_classes: int = 6      # 輸出分類
    input_channels: int = 1   # 輸入訊號的通道數
    embed_size: int = 192     # Transformer 的維度大小，原本是 192
    num_heads: int = 1        # MultiHeadAttention 的頭數，原本是 8
    expansion: int = 1        # MLP 的結構中，首先有一個線性層將輸入的維度擴展到 input_channels * expansion，


@dataclasses.dataclass()
class EcgConfig:
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    device: Union[int, str] = "cuda"
    lr: float = 0.000001 # 原本是 2e-4，因為調大 bath_size，所以調大 lr
                         # 1-400: 0.00005
                         # 400-444: 0.00001
                         # 444-: 0.000001
    num_epochs: int = 400
    validation_frequency: int = 2
