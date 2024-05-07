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
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
OBJECTS = ['C']
NUM_EPOCHS = 5000
INPUT_FEATURES = ['direct_radiation']
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
        historical_data, weather_data, TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT)
    merged_df, start_time_index = data_transformer.get_merged_df()
    X = merged_df[INPUT_FEATURES]
    y = merged_df[['value']]

    train_losses = []
    test_losses = []

    predictions = []
    ground_truth = []
    test_score = 0
    train_score = 0

    X_scaler = RobustScaler()
    y_scaler = RobustScaler()
    X_train, X_test, X_val, y_val, y_train, y_test = data_transformer.get_train_and_test_data(
        X, y)

    X_test_scaled = X_scaler.fit_transform(X_test)
    X_train_scaled = X_scaler.transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)

    y_train_scaled = y_scaler.fit_transform(
        y_train.values.reshape(-1, 1))

    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor = data_transformer.convert_train_test_data_to_tensors(
        X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled)

    backcast_length = X_train.shape[1]
    forecast_length = 1
    print(
        f'Backcast length: {backcast_length}; Forecast length: {forecast_length}')
    print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')

    model = NBeatsNet(
        device='cpu',
        stack_types=[NBeatsNet.GENERIC_BLOCK, NBeatsNet.SEASONALITY_BLOCK],
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
        object_name=object, patience=150, min_delta=0.001, model_type=ModelType.NBEATS)

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad()

        forecast, backcast = model(X_train_tensor)
        forecast = forecast.squeeze(1)
        train_loss = criterion(forecast, y_train_tensor)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_forecast, backcast = model(X_val_tensor)
            eval_forecast = eval_forecast.squeeze(1)
            test_loss = criterion(eval_forecast, y_val_tensor)

        eval_outputs = y_scaler.inverse_transform(
            eval_forecast.reshape(-1, 1))
        y_test_original = y_scaler.inverse_transform(
            y_val_tensor.reshape(-1, 1))

        r2 = r2_score(y_test_original, eval_outputs)
        if test_score < r2:
            test_score = r2

        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}, validation R2: {r2:.6f}")

        early_stopping.save_best_model_weights()
        early_stopping.update(test_loss, model)
        if early_stopping.should_stop():
            print(f'Early stopping at epoch {epoch}')
            break
    model.load_state_dict(torch.load(
        early_stopping.best_weights_path))

    model.eval()
    with torch.no_grad():
        forecast, backcast = model(X_test_tensor)
        test_forecast = forecast.squeeze(1)
        test_forecast = y_scaler.inverse_transform(
            test_forecast.reshape(-1, 1))
        y_test_original = y_scaler.inverse_transform(
            y_test_tensor.reshape(-1, 1))
        test_score = r2_score(y_test_original, test_forecast)
        ground_truth.extend(y_test_original)
        predictions.extend(test_forecast)

    rmse = mean_squared_error(ground_truth, predictions, squared=False)
    mae = mean_absolute_error(ground_truth, predictions)
    print(f'RMSE: {rmse}, MAE: {mae}')
    print(f'Test R2: {test_score}')

    # results_plot = PlotPredictions("nbeats", object_name=object, title=results_plot_title, save_path=results_save_path, ground_truth=ground_truth_df,
    #                                x_data=ground_truth, y_data=predictions)
    # results_plot.create_plot()
