import copy
import dataclasses
import datetime as dt


@dataclasses.dataclass
class ProductionDataPoint:
    object_name: str
    start_time: dt.datetime
    value: float
    _id: str

    def to_object(self):
        return {
            "start_time": self.start_time,
            "value": self.value,
        }


@dataclasses.dataclass
class WeatherDataPoint:
    object_name: str
    start_time: dt.datetime
    direct_radiation: float
    precipitation: float
    pressure: float
    rain: float
    relative_humidity: float
    temperature: float
    wind_speed: float
    _id: str

    def to_object(self):
        return {
            "start_time": self.start_time,
            "direct_radiation": self.direct_radiation,
            "precipitation": self.precipitation,
            "pressure": self.pressure,
            "rain": self.rain,
            "relative_humidity": self.relative_humidity,
            "temperature": self.temperature,
            "wind_speed": self.wind_speed,
        }


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_weights = None

    def update(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def should_stop(self):
        return self.early_stop
