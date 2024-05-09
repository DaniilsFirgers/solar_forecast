import copy
import dataclasses
import datetime as dt
import os
from matplotlib.dates import AutoDateLocator, DateFormatter
import matplotlib.pyplot as plt
from numpy import ndarray
from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from enum import Enum
from itertools import combinations
from torch.utils.data import Dataset
from typing import List, TypedDict
from sklearn.linear_model import Lasso, LinearRegression
import numpy as np


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
    NBEATS = 'NBEATS'
    GRU = 'GRU'


class EarlyStopping:
    def __init__(self, object_name: str, patience=10, min_delta=0, model_type=ModelType.LSTM):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_model_weights = None
        self.model_type = model_type
        self.save_dir = "trained_models/"
        self.object_name = object_name
        self.best_weights_path = None

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

    def save_best_model_weights(self):
        if self.best_model_weights is not None:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            self.best_weights_path = os.path.join(
                self.save_dir, f'best_{self.model_type.value}_weights_{self.object_name}.pt')
            torch.save(self.best_model_weights, self.best_weights_path)


class PlotLoss():
    def __init__(self, model_name: str, object_name: str, title: str, save_path: str, x_label='Apmācības zaudējumi', y_label='Validācijas zaudējumi', fig_size=(10, 5), x_data: list = [], y_data: list = [], x_color='blue', y_color='red'):
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
        # plt.title(
        #     self.title)
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_path)
        plt.close()


class PlotPredictions():
    def __init__(self, model_name: str, object_name: str, title: str, save_path: str, data: DataFrame | None, x_label='Patiesās vērtībās', y_label='Prognozetas vērtībās',  fig_size=(10, 5),  x_color='blue', y_color='green'):
        self.model_name = model_name
        self.object_name = object_name
        self.fig_size = fig_size
        self.x_label = x_label
        self.y_label = y_label
        self.title = title
        self.save_path = save_path
        self.x_color = x_color
        self.y_color = y_color
        self.data = data

    def create_plot(self):
        plt.figure(figsize=self.fig_size)
        plt.plot(self.data.index, self.data["value"],
                 label=self.x_label, color=self.x_color)
        plt.plot(self.data.index, self.data["predictions"],
                 label=self.y_label, color=self.y_color)
        plt.title(
            self.title)
        plt.legend()
        plt.grid(True)
        plt.gca().xaxis.set_major_locator(AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(DateFormatter('%d-%m-%y %H:%M'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        # plt.savefig(self.save_path)
        plt.close()


class DataTransformer:
    def __init__(self, historical_production_data: DataFrame, weather_data: DataFrame, test_ratio: int, train_ratio: int, valiation_ratio: int, random_state=42):
        self.historical_production_data = historical_production_data
        self.weather_data = weather_data
        self.merged_dataframe = None
        self.test_size = test_ratio
        self.train_size = train_ratio
        self.validation_size = valiation_ratio
        self.random_state = random_state

    def add_lagged_features(self, lagged_features: list, lag_steps: int):
        for feature in lagged_features:
            for i in range(1, lag_steps + 1):
                self.merged_dataframe[f'{feature}_lag_{i}'] = self.merged_dataframe[feature].shift(
                    i)

        # Drop rows with NaN values resulted from shifting
        self.merged_dataframe.dropna(inplace=True)

    def get_merged_df(self) -> tuple[DataFrame, DataFrame]:
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

        start_time_index = self.merged_dataframe.reset_index()[["start_time"]]
        self.merged_dataframe.reset_index(inplace=True)

        return self.merged_dataframe, start_time_index

    def get_train_and_test_data(self, X: ndarray, y: ndarray) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-self.train_size, shuffle=False,  random_state=self.random_state)

        X_val, X_test, y_val, y_test = train_test_split(
            X_test, y_test, test_size=self.test_size/(self.test_size + self.validation_size), shuffle=False,  random_state=self.random_state)

        return X_train, X_test, X_val, y_val, y_train, y_test

    def convert_train_test_data_to_tensors(self, X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        X_train = torch.from_numpy(X_train_scaled).float()
        y_train = torch.squeeze(torch.from_numpy(y_train_scaled).float())

        X_test = torch.from_numpy(X_test_scaled).float()
        y_test = torch.squeeze(torch.from_numpy(y_test_scaled).float())

        X_val = torch.from_numpy(X_val_scaled).float()
        y_val = torch.squeeze(torch.from_numpy(y_val_scaled).float())

        return X_train, X_val, X_test, y_train, y_val, y_test


ObjectConfig = dict[str, int]


class ModelWrapper(TypedDict):
    name: str
    model: Lasso | LinearRegression | None
    input_features: List[str]
    short_name: str
    hidden_layers: ObjectConfig | None
    layers: ObjectConfig | None
    dropout: float | None
    negative_slope: ObjectConfig | None
    patience: ObjectConfig | None


PARAMETERS_NAME_MAP = {"value": "Ražošanas vērtības", "temperature": "Temperatūra", "relative_humidity": "Relatīvais mitrums", "wind_speed": "Vēja ātrums", "pressure": "Atmosfēras spiediens",
                       "rain": "Lietus", "direct_radiation": "Tiešais starojums", "diffuse_radiation": "Difūzā atstarošanās", "precipitation": "Nokrišņi",
                       "direct_radiation_instant": "Momentālais tiešais starojums", "diffuse_radiation_instant": "Momentāla tieša difūzā atstarošanās", "direct_normal_irradiance": "Tiešais normālais starojums",
                       "direct_normal_irradiance_instant": "Mom. tiešais normālais starojums", "shortwave_radiation": "Īsviļņu radiācija", "terrestrial_radiation": "Zemes starojums", "terrestrial_radiation_instant": "Momentālais zemes starojums", "value_lag_1": "Ražošanas vērtības ar nobīdi 1"}


def generate_unique_feature_combinations(features):
    unique_combinations = []
    for r in range(1, len(features) + 1):
        for combo in combinations(features, r):
            sorted_combo = sorted(combo)
            if sorted_combo not in unique_combinations:
                unique_combinations.append(sorted_combo)
    return unique_combinations


def mean_bias_error(actual, predicted):
    """ Calculate the Mean Bias Error (MBE) between two arrays of predicted and actual values. """
    predicted = np.array(predicted)
    actual = np.array(actual)
    differences = predicted - actual
    mbe = np.mean(differences)
    return mbe


def adjusted_r_squared(r_squared, n, k):
    return 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))
