import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import torch.nn as nn
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import RobustScaler
import torch
from config.database import FORECAST_DB_NAME, PRODUCTION_COLLECTION_NAME, WEATHER_COLLECTION_NAME
from database.main import mongo_handler
import matplotlib.pyplot as plt
import matplotlib
from data_handling.transform import EarlyStopping, ModelType, PlotLoss, PlotPredictions, DataTransformer, ModelWrapper, mean_bias_error, adjusted_r_squared
from sklearn.linear_model import LinearRegression
from typing import List
from statsmodels.stats.stattools import durbin_watson
from models.main import GRU, LSTM, RNN
import joblib

plt.style.use('ggplot')
matplotlib.use('tkagg')
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.25
TEST_SPLIT = 0.05
OBJECTS = ['A', 'B']
NUM_EPOCHS = 5000
NN_NEED_TRAINING = False
LR_NEED_TRAINING = False


evaluation_data = []

MODELS: List[ModelWrapper] = [
    {"name": "GRU", "model": None, "input_features": ['shortwave_radiation', 'direct_radiation',
                                                      'relative_humidity', 'temperature', 'pressure', 'hour'], "short_name": "gru", "hidden_layers": {"A": 64, "B": 64, "C": 64}, "layers": {"A": 3, "B": 3, "C": 2}, "dropout": 0.1, "negative_slope": {"A": 1e-6, "B": 1e-6, "C": 1e-5}, "patience": {"A": 120, "B": 150, "C": 175}, "plot_color": "green"},
    {"name": "Lasso", "model": Lasso(alpha=0.001, max_iter=200, positive=True), "input_features": [
        'shortwave_radiation', 'direct_radiation',
        'relative_humidity', 'temperature', 'pressure', 'hour'], "short_name": "lasso", "hidden_layers": None, "layers": None, "dropout": None, "plot_color": "blue"},
    {"name": "Lineārā regresija", "model": LinearRegression(positive=True), "input_features": [
        'shortwave_radiation', 'relative_humidity', 'pressure', "rain", 'hour'], "short_name": "linear_regression", "hidden_layers": None, "layers": None, "dropout": None, "plot_color": "red"},
    {"name": "LSTM", "model": None, "input_features": ['shortwave_radiation', 'direct_radiation',
                                                       'relative_humidity', 'temperature', 'pressure', 'hour'], "short_name": "lstm", "hidden_layers": {"A": 128, "B": 128, "C": 64}, "layers": {"A": 3, "B": 2, "C": 2}, "dropout": 0.1, "negative_slope": {"A": 1e-6, "B": 1e-4, "C": 1e-6}, "patience": {"A": 100, "B": 120, "C": 175}, "plot_color": "magenta"},
    {"name": "RNN", "model": None, "input_features": ['shortwave_radiation', 'direct_radiation',
                                                      'relative_humidity', 'temperature', 'pressure', 'hour'], "short_name": "rnn", "hidden_layers": {"A": 64, "B": 128, "C": 128}, "layers": {"A": 3, "B": 3, "C": 2}, "dropout": 0.1, "negative_slope": {"A": 1e-6, "B": 1e-6, "C": 1e-7}, "patience": {"A": 150, "B": 150, "C": 175}, "plot_color": "yellow"},
    {"name": "Gradient Boosting", "model": GradientBoostingRegressor(n_estimators=750, learning_rate=0.01, alpha=0.9, n_iter_no_change=25), "input_features": ['shortwave_radiation', 'direct_radiation',
                                                                                                                                                               'relative_humidity', 'temperature', 'pressure', 'hour'], "short_name": "gb", "hidden_layers": None, "layers": None, "dropout": None, "plot_color": "cyan"}
]

for object in OBJECTS:
    list_of_predictions = []
    list_of_labels = []
    for model in MODELS:
        list_of_labels.append(model["name"])
        print(
            f'---------------------Object: {object}; Model: {model["name"]}---------------------')

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
        y = merged_df[['value']]
        X = merged_df[model["input_features"]]

        results_plot_title = f'Ražošanas prognozes pret patiesajām vērtībām {object} objektam - {model["name"]}'
        results_save_path = f'plots/{model["short_name"]}-{object}-results.png'

        train_losses = []
        eval_losses = []

        predictions = []
        ground_truth = []
        test_score = 0
        adjusted_val_test_score = 0
        predictions_df = None

        X_scaler = RobustScaler()
        y_scaler = RobustScaler()

        if model["short_name"] == "lasso" or model["short_name"] == "linear_regression" or model["short_name"] == "gb":

            X_train, X_test, X_val, y_val, y_train, y_test = data_transformer.get_train_and_test_data(
                X, y)

            X_test_scaled = X_scaler.fit_transform(X_test)
            X_train_scaled = X_scaler.transform(X_train)
            X_val_scaled = X_scaler.transform(X_val)

            y_train_scaled = y_scaler.fit_transform(
                y_train.values.reshape(-1, 1))
            y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1))
            y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

            lr_model = model["model"]

            if model["short_name"] == "gb":
                y_train_scaled = y_train_scaled.flatten()

            if LR_NEED_TRAINING:
                lr_model.fit(X_train_scaled, y_train_scaled)
                joblib.dump(
                    lr_model, f'trained_models/{model["short_name"]}_{object}.pkl')
            else:
                lr_model = joblib.load(
                    f'trained_models/{model["short_name"]}_{object}.pkl')

            lr_evals = lr_model.predict(X_val_scaled)

            score = lr_model.score(X_val_scaled, y_val_scaled)
            adjusted_score = adjusted_r_squared(
                test_score, X_val.shape[0], X_val.shape[1]
            )

            lr_evals = np.maximum(lr_evals, 0)

            lr_test = lr_model.predict(X_test_scaled)
            lr_test = np.maximum(lr_test, 0)
            test_score = lr_model.score(X_test_scaled, y_test_scaled)
            adjusted_test_score = adjusted_r_squared(
                test_score, X_test.shape[0], X_test.shape[1]
            )

            predictions = y_scaler.inverse_transform(
                lr_test.reshape(-1, 1))

            residuals = (y_test - predictions).values.flatten()

            durbin_watson_test_score = durbin_watson(residuals)
            print(f'Test Durbin-Watson score: {durbin_watson_test_score}')

            ground_truth = y_test

        elif model["short_name"] == "lstm" or model["short_name"] == "rnn" or model["short_name"] == "gru":
            loss_plot_title = f'{model["name"]} modeļa apmācības un validācijas zaudējumi {object} objektam'
            loss_save_path = f'plots/{model["name"]}-{object}-loss.png'

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

            hidden_layers = model["hidden_layers"][object]
            layers = model["layers"][object]
            dropout = model["dropout"]
            negative_slope = model["negative_slope"][object]
            patience = model["patience"][object]

            nn_model = model["model"]
            if model["short_name"] == "lstm":
                nn_model = LSTM(input_size=X_train_tensor.shape[1], hidden_size=hidden_layers,
                                num_layers=layers, output_size=1, dropout=dropout, negative_slope=negative_slope)
            elif model["short_name"] == "rnn":
                nn_model = RNN(input_size=X_train_tensor.shape[1], hidden_size=hidden_layers,
                               num_layers=layers, output_size=1, dropout=dropout, negative_slope=negative_slope)
            elif model["short_name"] == "gru":
                nn_model = GRU(input_dim=X_train_tensor.shape[1], hidden_dim=hidden_layers,
                               num_layers=layers, output_dim=1, droupout=dropout, negative_slope=negative_slope)

            criterion = nn.SmoothL1Loss()
            optimizer = torch.optim.Adam(
                nn_model.parameters(), lr=0.001)
            model_type: str
            if model["short_name"] == "lstm":
                model_type = ModelType.LSTM
            elif model["short_name"] == "rnn":
                model_type = ModelType.RNN
            elif model["short_name"] == "gru":
                model_type = ModelType.GRU

            early_stopping = EarlyStopping(
                object_name=object, patience=patience, min_delta=0.001, model_type=model_type)

            best_weght_path = f'trained_models/best_{model["name"]}_weights_{object}.pt'
            if NN_NEED_TRAINING:
                for epoch in range(NUM_EPOCHS):
                    outputs = nn_model(X_train_tensor.unsqueeze(1)).squeeze()
                    optimizer.zero_grad()
                    train_loss: torch.Tensor = criterion(
                        outputs, y_train_tensor)
                    train_loss.backward()

                    if model["short_name"] == "rnn":
                        torch.nn.utils.clip_grad_norm_(
                            nn_model.parameters(), 10.0)
                    optimizer.step()

                    nn_model.eval()
                    with torch.no_grad():
                        eval_outputs = nn_model(
                            X_val_tensor.unsqueeze(1)).squeeze()
                        eval_loss: torch.Tensor = criterion(
                            eval_outputs, y_val_tensor)

                    eval_outputs = y_scaler.inverse_transform(
                        eval_outputs.reshape(-1, 1)).flatten()
                    y_val_original = y_scaler.inverse_transform(
                        y_val_tensor.numpy().reshape(-1, 1)).flatten()

                    r2 = r2_score(y_val_original, eval_outputs)
                    adjusted_r2 = adjusted_r_squared(
                        r2, X_val.shape[0], X_val.shape[1])

                    if (epoch + 1) % 10 == 0:
                        print(
                            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss.item():.4f}, Test Loss: {eval_loss.item():.4f}, Validation R2: {r2:.4f}, Adjusted R2: {adjusted_r2:.4f}")

                    eval_losses.append(eval_loss.detach().numpy())
                    train_losses.append(train_loss.detach().numpy())

                    early_stopping.update(eval_loss.item(), nn_model)
                    if early_stopping.should_stop():
                        print(f'Early stopping at epoch {epoch}')
                        break

                early_stopping.save_best_model_weights()
                loss_plot = PlotLoss(model["name"], object_name=object, title=loss_plot_title, save_path=loss_save_path,
                                     x_data=train_losses, y_data=eval_losses)
                # TODO: save plot here
                loss_plot.create_plot()
                best_weght_path = early_stopping.best_weights_path

            nn_model.load_state_dict(torch.load(best_weght_path))
            nn_model.eval()
            with torch.no_grad():
                test_outputs = nn_model(X_test_tensor.unsqueeze(1)).squeeze()
                test_outputs = y_scaler.inverse_transform(
                    test_outputs.reshape(-1, 1)).flatten()
                y_val_original = y_scaler.inverse_transform(
                    y_test_tensor.numpy().reshape(-1, 1)).flatten()

                test_score = r2_score(y_val_original, test_outputs)
                adjusted_r2 = adjusted_r_squared(
                    test_score, X_test.shape[0], X_test.shape[1])

                predictions = test_outputs
                ground_truth = y_val_original

        predictions_df = pd.DataFrame(
            {'predictions': predictions.flatten()}, index=X_test.index)

        predictions_df = predictions_df.join(
            start_time_index
        )
        predictions_df.sort_values(by='start_time', inplace=True)
        predictions_df.set_index('start_time', inplace=True)

        list_of_predictions.append(predictions_df)

        rmse = mean_squared_error(ground_truth, predictions, squared=False)
        mae = mean_absolute_error(ground_truth, predictions)
        mbe = mean_bias_error(ground_truth, predictions)
        print(
            f'R^2: {test_score}, RMSE: {rmse}, MAE: {mae}, MBE: {mbe}'
        )
        evaluation_data.append({
            'Model': model["name"],
            'Solar Park': object,
            'R^2': test_score,
            'MAE': mae,
            'RMSE': rmse,
            'MBE': mbe
        })

    truth_values = y_test.join(
        start_time_index
    )

    truth_values.sort_values(by='start_time', inplace=True)
    truth_values.set_index('start_time', inplace=True)
    results_plot = PlotPredictions(model["name"], object_name=object, save_path=results_save_path,
                                   predictions=list_of_predictions, truth_values=truth_values, labels=list_of_labels)
    results_plot.create_plot()

evaluation_df = pd.DataFrame(evaluation_data)


pivot_df = evaluation_df.pivot(index='Model', columns='Solar Park')

# Flatten column names
pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]

# Reset index
pivot_df.reset_index(inplace=True)


models = pivot_df['Model']
metrics = ['R^2', 'MAE', 'RMSE', 'MBE']

bar_width = 0.3

for metric in metrics:
    metric_data_A = pivot_df[f'{metric}_A']
    metric_data_B = pivot_df[f'{metric}_B']

    # Calculate positions for each set of bars
    x = np.arange(len(models))  # positions for the model groups
    group_width = 3 * bar_width  # The total width of a group (three bars)
    inter_group_margin = 0.5  # Space between groups

    plt.figure(figsize=(10, 6))
    bars_a = plt.bar(x, metric_data_A, width=bar_width,
                     label='Saules parks “A”')
    bars_b = plt.bar(x + bar_width, metric_data_B,
                     width=bar_width, label='Saules parks “B”')
    # Set labels, titles, and legends
    plt.xlabel('Model')
    plt.ylabel(metric)
    # plt.title(f'{metric} pēc modeļa un saules parka')
    # Centering ticks under the middle bar of each group
    plt.xticks(x + group_width / 2 - bar_width / 2, models)
    plt.legend()

    for bars in [bars_a, bars_b]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval,
                     round(yval, 3), ha='center', va='bottom')

    # Optional: Set x-axis limits to add some padding for clarity
    plt.xlim(-0.5, len(models) - 1 + group_width + inter_group_margin)

    plt.show()
