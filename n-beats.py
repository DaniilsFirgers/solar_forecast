import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import torch
from config.database import FORECAST_DB_NAME, PRODUCTION_COLLECTION_NAME, WEATHER_COLLECTION_NAME
from database.main import mongo_handler
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from data_handling.transform import EarlyStopping, ModelType, PlotLoss, PlotPredictions, DataTransformer, ModelWrapper
from typing import List
from nbeats_pytorch.model import NBeatsNet

plt.style.use('ggplot')
matplotlib.use('tkagg')
SPLIT_RATIO = 0.3
OBJECTS = ['A']
NUM_EPOCHS = 1500
INPUT_FEATURES = ['direct_radiaation']
for object in OBJECTS:
    results_plot_title = f'Ražošanas prognozes pret patiesajām vērtībām {object} objektam - nbeats'
    results_save_path = f'plots/n-beats-{object}-results.png'

    print(
        f'---------------------Object: {object}; Model: Nbeats---------------------')

    filter_query = {
        "object_name": object
    }

    historical_data = mongo_handler.retrieve_production_data(
        FORECAST_DB_NAME, PRODUCTION_COLLECTION_NAME, filter_query)
    weather_data = mongo_handler.retrieve_weather_data(
        FORECAST_DB_NAME, WEATHER_COLLECTION_NAME, filter_query)
    if historical_data is None or weather_data is None:
        print("Error retrieving data from MongoDB.")
        exit(1)

    data_transformer = DataTransformer(
        historical_data, weather_data, 1-SPLIT_RATIO)

    merged_df = data_transformer.get_merged_df()
    X = merged_df[INPUT_FEATURES]
    y = merged_df['value']

    print(f'X: {X.shape}, y: {y.shape}')

    X_scaler = RobustScaler()
    X_scaled = X_scaler.fit_transform(X)

    y_scaler = RobustScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    y_scaler.fit(y.values.reshape(-1, 1))
    y_scaled = y_scaler.transform(y.values.reshape(-1, 1))

    split_index = int(len(X_scaled) * (1 - SPLIT_RATIO))
    ground_truth_df = merged_df['value'].iloc[split_index:]

    train_losses = []
    test_losses = []

    predictions = []
    ground_truth = []
    test_score = 0
    train_score = 0

    X_train, X_test, y_train, y_test, ground_truth_df = data_transformer.get_train_and_test(
        X_scaled, y_scaled)

    backcast_length = X_train.shape[1]
    forecast_length = 1
    print(
        f'Backcast length: {backcast_length}; Forecast length: {forecast_length}')
    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')

    model = NBeatsNet(
        device='cpu',
        stack_types=[NBeatsNet.GENERIC_BLOCK],
        forecast_length=forecast_length,
        backcast_length=backcast_length,
        # thetas_dim=[7, 8],
        nb_blocks_per_stack=1,
        share_weights_in_stack=False,
        hidden_layer_units=1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    early_stopping = EarlyStopping(
        object_name=object, patience=40, min_delta=0.01, model_type=ModelType.NBEATS)

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()

        forecast, backcast = model(X_train)
        forecast = forecast.squeeze(1)
        train_loss = criterion(forecast, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_forecast, backcast = model(X_test)
            test_forecast = test_forecast.squeeze(1)
            test_loss = criterion(test_forecast, y_test)

        test_outputs = y_scaler.inverse_transform(
            test_forecast.reshape(-1, 1))
        y_test_original = y_scaler.inverse_transform(
            y_test.reshape(-1, 1))

        r2 = r2_score(y_test_original, test_outputs)
        if test_score < r2:
            test_score = r2

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}, Test R2: {r2:.6f}")

        early_stopping.update(test_loss, model)
        if early_stopping.should_stop():
            predictions = test_outputs
            ground_truth = y_test_original
            print(f'Early stopping at epoch {epoch}')
            break

    rmse = mean_squared_error(ground_truth, predictions, squared=False)
    mae = mean_absolute_error(ground_truth, predictions)
    print(f'RMSE: {rmse}, MAE: {mae}')
    results_plot = PlotPredictions("nbeats", object_name=object, title=results_plot_title, save_path=results_save_path, ground_truth=ground_truth_df,
                                   x_data=ground_truth, y_data=predictions)
    results_plot.create_plot()
