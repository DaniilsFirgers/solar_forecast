import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split
from data_handling.transform import EarlyStopping, ModelType, PlotLoss, PlotPredictions, DataTransformer, ModelWrapper
from sklearn.linear_model import LinearRegression
from typing import List
from statsmodels.stats.stattools import durbin_watson
from models.main import GRU, LSTM, RNN

plt.style.use('ggplot')
matplotlib.use('tkagg')
SPLIT_RATIO = 0.80
OBJECTS = ['A', 'B', 'C']
NUM_EPOCHS = 15000


evaluation_data = []

MODELS: List[ModelWrapper] = [
    # {"name": "GRU", "model": None, "input_features": ['shortwave_radiation',
    #                                                   'pressure', 'relative_humidity', 'temperature', 'rain', 'month', 'day_of_week', 'hour'], "short_name": "gru", "hidden_layers": 128, "layers": 1},
    {"name": "Lasso", "model": Lasso(alpha=0.01, max_iter=1000, positive=True), "input_features": [
        'shortwave_radiation',
        'relative_humidity', 'month', 'day_of_week', 'hour'], "short_name": "lasso", "hidden_layers": None, "layers": None},
    {"name": "Lineārā regresija", "model": LinearRegression(positive=True), "input_features": [
        'shortwave_radiation', 'relative_humidity', 'pressure', "rain", 'month', 'day_of_week', 'hour'], "short_name": "linear_regression", "hidden_layers": None, "layers": None},
    # {"name": "LSTM", "model": None, "input_features": ['direct_radiation', 'pressure', 'relative_humidity',
    #                                                    'temperature', 'terrestrial_radiation', 'wind_speed', 'month', 'day_of_week', 'hour'], "short_name": "lstm", "hidden_layers": 128, "layers": 2},
    # {"name": "RNN", "model": None, "input_features": ['pressure', 'rain', 'relative_humidity', 'shortwave_radiation',
    #                                                   'temperature', 'terrestrial_radiation', 'wind_speed', 'month', 'day_of_week', 'hour'], "short_name": "rnn", "hidden_layers": 128, "layers": 2},
]

for model in MODELS:
    for object in OBJECTS:

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

        data_transformer = DataTransformer(historical_data, weather_data)

        merged_df = data_transformer.get_merged_df()
        X = merged_df[model["input_features"]]
        y = merged_df['value']

        X_scaler = RobustScaler()
        X_scaled = X_scaler.fit_transform(X)

        y_scaler = RobustScaler()
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

        y_scaler.fit(y.values.reshape(-1, 1))
        y_scaled = y_scaler.transform(y.values.reshape(-1, 1))

        split_index = int(len(X_scaled) * (1 - SPLIT_RATIO))
        ground_truth_df = merged_df['value'].iloc[split_index:]

        results_plot_title = f'Ražošanas prognozes pret patiesajām vērtībām {object} objektam - {model["name"]}'
        results_save_path = f'plots/{model["short_name"]}-{object}-results.png'

        train_losses = []
        test_losses = []

        predictions = []
        ground_truth = []
        test_score = 0
        train_score = 0

        if model["short_name"] == "lasso" or model["short_name"] == "linear_regression":

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=SPLIT_RATIO, random_state=42)
            lr_model = model["model"]

            lr_model.fit(X_train, y_train)
            train_score = lr_model.score(X_train, y_train)
            test_score = lr_model.score(X_test, y_test)
            lr_preds = lr_model.predict(X_test)
            lr_preds = np.maximum(lr_preds, 0)
            residuals = (y_test - lr_preds).flatten()
            durbin_watson_score = durbin_watson(residuals)
            print(f'Durbin-Watson score: {durbin_watson_score}')

            predictions = y_scaler.inverse_transform(
                lr_preds.reshape(-1, 1))
            ground_truth = y_scaler.inverse_transform(
                y_test.reshape(-1, 1))

        elif model["short_name"] == "lstm" or model["short_name"] == "rnn" or model["short_name"] == "gru":
            loss_plot_title = f'{model["name"]} modeļa apmācības un validācijas zaudējumi {object} objektam'
            loss_save_path = f'plots/{model["name"]}-{object}-loss.png'

            X_train, X_test, y_train, y_test, ground_truth_df = data_transformer.get_train_and_test(
                X_scaled, y_scaled)

            nn_model = model["model"]
            if model["short_name"] == "lstm":
                nn_model = LSTM(input_size=X_train.shape[1], hidden_size=model["hidden_layers"],
                                num_layers=model["layers"], output_size=1)
            elif model["short_name"] == "rnn":
                nn_model = RNN(input_size=X_train.shape[1], hidden_size=model["hidden_layers"],
                               num_layers=model["layers"], output_size=1)
            elif model["short_name"] == "gru":
                nn_model = GRU(input_dim=X_train.shape[1], hidden_dim=model["hidden_layers"],
                               num_layers=model["layers"], output_dim=1)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
            model_type = ModelType.LSTM if model["short_name"] == "lstm" else ModelType.RNN

            patience = 50

            early_stopping = EarlyStopping(
                object_name=object, patience=50, min_delta=0.0001, model_type=model_type)

            for epoch in range(NUM_EPOCHS):
                outputs = nn_model(X_train.unsqueeze(1)).squeeze()
                optimizer.zero_grad()
                train_loss: torch.Tensor = criterion(outputs, y_train)
                train_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    test_outputs = nn_model(X_test.unsqueeze(1)).squeeze()
                    test_loss: torch.Tensor = criterion(test_outputs, y_test)

                test_outputs = y_scaler.inverse_transform(
                    test_outputs.reshape(-1, 1)).flatten()
                y_test_original = y_scaler.inverse_transform(
                    y_test.numpy().reshape(-1, 1)).flatten()

                r2 = r2_score(y_test_original, test_outputs)
                if test_score < r2:
                    test_score = r2

                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test R2: {r2:.4f}")

                test_losses.append(test_loss.detach().numpy())
                train_losses.append(train_loss.detach().numpy())

                early_stopping.update(test_loss, nn_model)
                if early_stopping.should_stop():
                    predictions = test_outputs
                    ground_truth = y_test_original
                    print(f'Early stopping at epoch {epoch}')
                    break

            early_stopping.save_best_model_weights()
            loss_plot = PlotLoss(model["name"], object_name=object, title=loss_plot_title, save_path=loss_save_path,
                                 x_data=train_losses, y_data=test_losses)
            # TODO: save plot here
            # loss_plot.create_plot()

        results_plot = PlotPredictions(model["name"], object_name=object, title=results_plot_title, save_path=results_save_path, ground_truth=ground_truth_df,
                                       x_data=ground_truth, y_data=predictions)

        rmse = mean_squared_error(ground_truth, predictions, squared=False)
        mae = mean_absolute_error(ground_truth, predictions)
        evaluation_data.append({
            'Model': model["name"],
            'Solar Park': object,
            'R^2': test_score,
            'MAE': mae,
            'RMSE': rmse
        })

        print("Test R^2:", test_score)
        print("Train R^2:", train_score)
        print("Mean Squared Error:", rmse)
        print("Mean Absolute Error:", mae)
        # results_plot.create_plot()


evaluation_df = pd.DataFrame(evaluation_data)

# Now, you have all your evaluation metrics stored in the DataFrame
print(evaluation_df)

pivot_df = evaluation_df.pivot(index='Model', columns='Solar Park')

# Flatten column names
pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]

# Reset index
pivot_df.reset_index(inplace=True)

print(pivot_df)

models = pivot_df['Model']
metrics = ['R^2', 'MAE', 'RMSE']

bar_width = 0.3

for metric in metrics:
    metric_data_A = pivot_df[f'{metric}_A']
    metric_data_B = pivot_df[f'{metric}_B']
    metric_data_C = pivot_df[f'{metric}_C']

    # Calculate positions for each set of bars
    x = np.arange(len(models))  # positions for the model groups
    group_width = 3 * bar_width  # The total width of a group (three bars)
    inter_group_margin = 0.5  # Space between groups

    plt.figure(figsize=(10, 6))
    plt.bar(x, metric_data_A, width=bar_width, label='Saules parks “A”')
    plt.bar(x + bar_width, metric_data_B,
            width=bar_width, label='Saules parks “B”')
    plt.bar(x + 2 * bar_width, metric_data_C,
            width=bar_width, label='Saules parks “C”')

    # Set labels, titles, and legends
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.title(f'{metric} pēc modeļa un saules parka')
    # Centering ticks under the middle bar of each group
    plt.xticks(x + group_width / 2 - bar_width / 2, models)
    plt.legend()

    # Optional: Set x-axis limits to add some padding for clarity
    plt.xlim(-0.5, len(models) - 1 + group_width + inter_group_margin)

    plt.show()
