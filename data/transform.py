import dataclasses
import datetime as dt
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
from torch.utils.data import Dataset


@dataclasses.dataclass
class Datapoint:
    start_time: dt.datetime
    production: float
    interval: int
    area: str
    reference: str
    internal_type: str
    object_type: str
    _id: str

    def to_object(self):
        return {
            "start_time": self.start_time,
            "production": self.production,
        }


class CustomDataset(Dataset):
    def __init__(self, data: DataFrame):
        self.data = data
        self.scaler_consumption = MinMaxScaler()
        self.scaler_consumption.fit_transform(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
