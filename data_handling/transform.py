import copy
import dataclasses
import datetime as dt
import matplotlib.pyplot as plt
from numpy import ndarray
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch


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


class DataTransformer:
    def __init__(self, historical_production_data: DataFrame, weather_data: DataFrame, test_size=0.25, random_state=42):
        self.historical_production_data = historical_production_data
        self.weather_data = weather_data
        self.merged_dataframe = None
        self.test_size = test_size
        self.random_state = random_state

    def get_train_test_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.historical_production_data['start_time'] = pd.to_datetime(
            self.historical_production_data['start_time'])
        self.weather_data['start_time'] = pd.to_datetime(
            self.weather_data['start_time'])

        self.merged_dataframe = pd.merge(self.historical_production_data, self.weather_data,
                                         on='start_time', how='inner')
        self.merged_dataframe.set_index("start_time", inplace=True)

        scaled_data = self.__scale_data()
        return self.__get_train_and_test(scaled_data)

    def __scale_data(self) -> tuple[ndarray, ndarray]:
        X = self.merged_dataframe[['shortwave_radiation',
                                   'temperature', 'terrestrial_radiation_instant', 'relative_humidity', 'pressure']]
        y = self.merged_dataframe['value']

        scaler = MinMaxScaler()

        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

        return X_scaled, y_scaled

    def __get_train_and_test(self, scaled_data: tuple[ndarray, ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_scaled, y_scaled = scaled_data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=self.test_size,  random_state=self.random_state)

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.squeeze(torch.from_numpy(y_train).float())

        X_test = torch.from_numpy(X_test).float()
        y_test = torch.squeeze(torch.from_numpy(y_test).float())

        return X_train, X_test, y_train, y_test
