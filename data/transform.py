import dataclasses
import datetime as dt
from typing import Any, Dict
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
import torch
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
    def __init__(self, data: np.ndarray[Any, np.dtype]):
        self.data = data
        # Use scikit-learn MinMaxScaler
        self.scaler_consumption = MinMaxScaler()
        self.scaler_consumption.fit(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        consumption_value = self.scaler_consumption.transform(
            [self.data[idx].flatten()])
        return np.squeeze(consumption_value)
