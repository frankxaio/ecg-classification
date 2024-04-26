import dataclasses
from enum import Enum
from typing import Dict, Union

from ecg_tools.augmentations import Compose, RandomNoise, RandomShift


class Mode(Enum):
    train = "train"
    eval = "eval"


@dataclasses.dataclass()
class DatasetConfig:
    batch_size: int = 32
    num_workers: int = 8  # 多線程設定，設成0不會出錯，但是會很慢
    path: Dict = dataclasses.field(default_factory=lambda: {
        Mode.train: "..\\data\\mitbih_ptbdb_train_sliced.csv",
        Mode.eval: "..\\data\\mitbih_ptbdb_test_sliced.csv"
    })
    transforms: Dict = dataclasses.field(default_factory=lambda: {
        Mode.train: Compose([RandomNoise(0.05, 0.5),
                             RandomShift(10, 0.5)]),
        Mode.eval: lambda x: x})


@dataclasses.dataclass()
class ModelConfig:
    num_layers: int = 6
    signal_length: int = 149  # 原本是 187，縮減訊號長度成 149
    num_classes: int = 6
    input_channels: int = 1
    embed_size: int = 192
    num_heads: int = 8
    expansion: int = 4


@dataclasses.dataclass()
class EcgConfig:
    dataset: DatasetConfig = DatasetConfig()
    model: ModelConfig = ModelConfig()
    device: Union[int, str] = "cuda"
    lr: float = 2e-4
    num_epochs: int = 400
    validation_frequency: int = 2
