import copy
import dataclasses
import datetime as dt
import matplotlib.pyplot as plt


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
    diffuse_radiation: float
    direct_normal_irradiance: float
    direct_normal_irradiance_instant: float
    direct_radiation_instant: float
    shortwave_radiation: float
    terrestrial_radiation: float
    terrestrial_radiation_instant: float
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
            "diffuse_radiation": self.diffuse_radiation,
            "direct_normal_irradiance": self.direct_normal_irradiance,
            "direct_normal_irradiance_instant": self.direct_normal_irradiance_instant,
            "direct_radiation_instant": self.direct_radiation_instant,
            "shortwave_radiation": self.shortwave_radiation,
            "terrestrial_radiation": self.terrestrial_radiation,
            "terrestrial_radiation_instant": self.terrestrial_radiation_instant,
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


class Plot():
    def __init__(self, model_name: str, object_name: str, fig_size=(10, 5), train_losses: list = [], test_losses: list = []):
        self.model_name = model_name
        self.object_name = object_name
        self.fig_size = fig_size
        self.train_losses = train_losses
        self.test_losses = test_losses

    def plot_model_results(self):
        plt.figure(figsize=self.fig_size)
        plt.plot(range(1, len(self.train_losses) + 1),
                 self.train_losses, label='Training Loss')
        plt.plot(range(1, len(self.test_losses) + 1),
                 self.test_losses, label='Validation Loss')
        plt.title(
            f'Training and Validation Loss for {self.model_name} model - object {self.object_name}')
        plt.legend()
        plt.savefig(f'plots/{self.model_name}-{self.object_name}.png')
        plt.show()
