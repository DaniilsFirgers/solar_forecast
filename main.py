import torch
import torch.nn as nn
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from config.database import PRODUCTION_HISTORY_DB_NAME, PRODUCTION_HISTORY_COLLECTION_NAME, PRODUCTION_WEATHER_COLLECTION_NAME
from database.main import mongo_handler
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from data_handling.transform import EarlyStopping, Plot, DataTransformer
from models.main import LSTM
from sklearn.metrics import r2_score

plt.style.use('ggplot')
matplotlib.use('tkagg')
SPLIT_RATIO = 0.75
NUM_EPOCHS = 1000
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 1
OBJECTS = ['A', 'B', 'C']

scaler = MinMaxScaler()

for object in OBJECTS:

    filter_query = {
        "object_name": object
    }

    historical_data = mongo_handler.retrieve_production_data(
        PRODUCTION_HISTORY_DB_NAME, PRODUCTION_HISTORY_COLLECTION_NAME, filter_query)
    weather_data = mongo_handler.retrieve_weather_data(
        PRODUCTION_HISTORY_DB_NAME, PRODUCTION_WEATHER_COLLECTION_NAME, filter_query)
    if historical_data is None or weather_data is None:
        print("Error retrieving data from MongoDB.")
        exit(1)

    data_transformer = DataTransformer(historical_data, weather_data)

    merged_df = data_transformer.get_merged_df()

    X = merged_df[['shortwave_radiation',
                   'temperature', 'terrestrial_radiation_instant', 'relative_humidity', 'pressure']]
    y = merged_df['value']

    X_scaled = scaler.fit_transform(X)
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = data_transformer.get_train_and_test(
        X_scaled, y_scaled)

    lstm_model = LSTM(input_size=X_train.shape[1], hidden_size=LSTM_HIDDEN_SIZE,
                      num_layers=LSTM_LAYERS, output_size=1)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.002)

    train_losses = []
    test_losses = []

    early_stopping = EarlyStopping(patience=50, min_delta=0.0001)
    plot = Plot('LSTM', object_name=object, fig_size=(10, 5),
                train_losses=train_losses, test_losses=test_losses)

    BEST_VAL_LOSS = int(1e9)
    # Training loop
    for epoch in range(NUM_EPOCHS):
        outputs = lstm_model(X_train.unsqueeze(1)).squeeze()
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        running_vloss = 0.0

        with torch.no_grad():
            test_outputs = lstm_model(X_test.unsqueeze(1)).squeeze()
            test_loss = criterion(test_outputs, y_test)

        if (epoch + 1) % 10 == 0:
            test_outputs = scaler.inverse_transform(
                test_outputs.reshape(-1, 1)).flatten()
            y_test_original = scaler.inverse_transform(
                y_test.numpy().reshape(-1, 1)).flatten()
            r2 = r2_score(y_test_original, test_outputs)
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test R2: {r2:.4f}")

        test_losses.append(test_loss)
        early_stopping.update(test_loss, lstm_model)
        if early_stopping.should_stop():
            print(f'Early stopping at epoch {epoch}')
            break

    plot.plot_model_results()
