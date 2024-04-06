import copy
import dataclasses
import datetime as dt
import matplotlib.pyplot as plt
from numpy import ndarray
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from enum import Enum


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


class ModelType(Enum):
    LSTM = 'LSTM'
    RNN = 'RNN'


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, model_type=ModelType.LSTM):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_weights = None
        self.model_type = model_type

    def update(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            if self.model_type == ModelType.LSTM:
                self.best_model_weights = copy.deepcopy(model.state_dict())
            elif self.model_type == ModelType.RNN:
                self.best_model_weights = copy.deepcopy(
                    model.get_weights())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def should_stop(self):
        return self.early_stop


class Plot():
    def __init__(self, model_name: str, object_name: str, x_label: str, y_label: str, title: str, save_path: str, fig_size=(10, 5), x_data: list = [], y_data: list = [], x_color='blue', y_color='red'):
        self.model_name = model_name
        self.object_name = object_name
        self.fig_size = fig_size
        self.x_data = x_data
        self.y_data = y_data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.save_path = save_path
        self.x_color = x_color
        self.y_color = y_color

    def create_plot(self):
        plt.figure(figsize=self.fig_size)
        plt.plot(range(1, len(self.x_data) + 1), self.x_data,
                 label=self.x_label, color=self.x_color)
        plt.plot(range(1, len(self.y_data) + 1), self.y_data,
                 label=self.y_label, color=self.y_color)
        plt.title(
            self.title)
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path)
        plt.close()


class DataTransformer:
    def __init__(self, historical_production_data: DataFrame, weather_data: DataFrame, test_size=0.25, random_state=42):
        self.historical_production_data = historical_production_data
        self.weather_data = weather_data
        self.merged_dataframe = None
        self.test_size = test_size
        self.random_state = random_state

    def add_lagged_features(self, lagged_features: list, lag_steps: int):
        for feature in lagged_features:
            for i in range(1, lag_steps + 1):
                self.merged_dataframe[f'{feature}_lag_{i}'] = self.merged_dataframe[feature].shift(
                    i)

        # Drop rows with NaN values resulted from shifting
        self.merged_dataframe.dropna(inplace=True)

    def get_merged_df(self) -> DataFrame:
        self.historical_production_data['start_time'] = pd.to_datetime(
            self.historical_production_data['start_time'])
        self.weather_data['start_time'] = pd.to_datetime(
            self.weather_data['start_time'])

        self.merged_dataframe = pd.merge(self.historical_production_data, self.weather_data,
                                         on='start_time', how='inner')

        self.merged_dataframe['year'] = self.merged_dataframe['start_time'].dt.year
        self.merged_dataframe['month'] = self.merged_dataframe['start_time'].dt.month
        self.merged_dataframe['day_of_week'] = self.merged_dataframe['start_time'].dt.dayofweek
        self.merged_dataframe['hour'] = self.merged_dataframe['start_time'].dt.hour

        self.merged_dataframe.set_index("start_time", inplace=True)

        return self.merged_dataframe

    def get_train_and_test(self, X_scaled: ndarray, y_scaled: ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=self.test_size,  random_state=self.random_state)

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.squeeze(torch.from_numpy(y_train).float())

        X_test = torch.from_numpy(X_test).float()
        y_test = torch.squeeze(torch.from_numpy(y_test).float())

        return X_train, X_test, y_train, y_test
