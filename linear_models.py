import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from config.database import FORECAST_DB_NAME, PRODUCTION_COLLECTION_NAME, WEATHER_COLLECTION_NAME
from database.main import mongo_handler
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from data_handling.transform import PlotPredictions, DataTransformer, LinearModel
from sklearn.linear_model import LinearRegression
from typing import List

plt.style.use('ggplot')
matplotlib.use('tkagg')
SPLIT_RATIO = 0.70
OBJECTS = ['A']
INPUT_FEATURES = ['direct_radiation', 'value_lag_1']
LAGGED_FEATURES = ['value']
LAG_STEPS = 1

LINEAR_MODELS: List[LinearModel] = [
    {"name": "Lasso", "model": Lasso(alpha=0.01), "input_features": [
        'direct_radiation', 'value_lag_1'], "lagged_features": ['value'], "lag_steps": 1, "short_name": "lasso"},
    {"name": "Lineārā regresija", "model": LinearRegression(), "input_features": [
        'direct_radiation', 'value_lag_1'], "lagged_features": ['value'], "lag_steps": 1, "short_name": "linear_regression"}
]

for lr_model in LINEAR_MODELS:
    for object in OBJECTS:

        print(
            f'---------------------Object: {object}; Model: {lr_model["name"]}---------------------')

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
        if lr_model["lagged_features"]:
            data_transformer.add_lagged_features(
                lr_model["lagged_features"], lr_model["lag_steps"])
        X = merged_df[lr_model["input_features"]]
        y = merged_df['value']

        X_scaler = MinMaxScaler()
        X_scaled = X_scaler.fit_transform(X)

        y_scaler = MinMaxScaler()
        y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=SPLIT_RATIO, random_state=17)

        split_index = int(len(X_scaled) * (1 - SPLIT_RATIO))
        ground_truth_df = merged_df['value'].iloc[split_index:]

        results_plot_title = f'Ražošanas prognozes pret patiesajām vērtībām {object} objektam - {lr_model["name"]}'
        results_save_path = f'plots/{lr_model["short_name"]}-{object}-results.png'
        model = lr_model["model"]

        model.fit(X_train, y_train)
        train_score_ls = model.score(X_train, y_train)
        test_score_ls = model.score(X_test, y_test)

        y_train_pred = model.predict(X_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)

        predictions = y_scaler.inverse_transform(
            predictions.reshape(-1, 1))
        ground_truth = y_scaler.inverse_transform(
            y_test.reshape(-1, 1))

        results_plot = PlotPredictions(lr_model["name"], object_name=object, title=results_plot_title, save_path=results_save_path, ground_truth=ground_truth_df,
                                       x_data=ground_truth, y_data=predictions)

        mse = mean_squared_error(ground_truth, predictions)

        print("Test R^2:", test_score_ls)
        print("Train R^2:", train_score_ls)
        print("Mean Squared Error:", np.sqrt(mse))
        results_plot.create_plot()
