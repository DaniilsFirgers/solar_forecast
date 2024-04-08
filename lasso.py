from sklearn.linear_model import Lasso
import torch
import torch.nn as nn
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from config.database import FORECAST_DB_NAME, PRODUCTION_COLLECTION_NAME, WEATHER_COLLECTION_NAME
from database.main import mongo_handler
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from data_handling.transform import EarlyStopping, Plot, DataTransformer
from models.main import LSTM
from sklearn.metrics import mean_squared_error, r2_score

plt.style.use('ggplot')
matplotlib.use('tkagg')
SPLIT_RATIO = 0.75
NUM_EPOCHS = 1000
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 1
OBJECTS = ['B']
INPUT_FEATURES = ['temperature', 'relative_humidity', 'pressure', 'rain',
                  'wind_speed', "shortwave_radiation", 'month', 'day_of_week', 'hour']
LAGGED_FEATURES = ['value']
LAG_STEPS = 1

for object in OBJECTS:

    print(f'---------------------Object: {object}---------------------')

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
    # data_transformer.add_lagged_features(LAGGED_FEATURES, LAG_STEPS)
    X = merged_df[INPUT_FEATURES]
    y = merged_df['value']

    X_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)

    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=SPLIT_RATIO, random_state=42)

    # Define the Lasso regression model
    lasso_model = Lasso(alpha=0.001)  # Adjust alpha as needed

    # Train the Lasso model
    lasso_model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = lasso_model.predict(X_train)
    y_test_pred = lasso_model.predict(X_test)

    # Reverse scaling for predictions
    y_train_pred_original = y_scaler.inverse_transform(
        y_train_pred.reshape(-1, 1))
    y_test_pred_original = y_scaler.inverse_transform(
        y_test_pred.reshape(-1, 1))
    y_train_original = y_scaler.inverse_transform(
        y_train.reshape(-1, 1))
    y_test_original = y_scaler.inverse_transform(
        y_test.reshape(-1, 1))

    # Evaluate the model
    train_rmse = mean_squared_error(
        y_train_original, y_train_pred_original, squared=False)
    test_rmse = mean_squared_error(
        y_test_original, y_test_pred_original, squared=False)
    train_r2 = r2_score(y_train_original, y_train_pred_original)
    test_r2 = r2_score(y_test_original, y_test_pred_original)

    print("Train RMSE:", train_rmse)
    print("Test RMSE:", test_rmse)
    print("Train R^2:", train_r2)
    print("Test R^2:", test_r2)
