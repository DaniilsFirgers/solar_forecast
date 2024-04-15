from time import sleep
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from config.database import FORECAST_DB_NAME, PRODUCTION_COLLECTION_NAME, WEATHER_COLLECTION_NAME
from database.main import mongo_handler
import matplotlib.pyplot as plt
import matplotlib
from data_handling.transform import EarlyStopping, DataTransformer, generate_unique_feature_combinations
from models.main import LSTM, RNN
from sklearn.metrics import r2_score
import logging

logging.basicConfig(filename="feature_selection/lstm_features.log", level=logging.INFO,
                    format="%(asctime)s:%(levelname)s:%(message)s", filemode='w')

plt.style.use('ggplot')
matplotlib.use('tkagg')
SPLIT_RATIO = 0.75
NUM_EPOCHS = 1000
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 1
TEST_OBJECT = 'C'
INPUT_FEATURES = ['temperature',  'relative_humidity', 'pressure', 'rain',
                  'wind_speed', "shortwave_radiation", 'direct_normal_irradiance', 'direct_normal_irradiance_instant', 'direct_radiation', 'terrestrial_radiation']

FEATURE_COMBINATIONS = generate_unique_feature_combinations(INPUT_FEATURES)

filter_query = {
    "object_name": TEST_OBJECT
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

best_combination = {"r_squared": 0, "features": []}

for inx, features in enumerate(FEATURE_COMBINATIONS):
    print(f"Testing combination {inx+1}/{len(FEATURE_COMBINATIONS)}")
    count = 0

    X = merged_df[features]
    y = merged_df['value']

    X_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)

    Y_scaler = MinMaxScaler()
    Y_scaler.fit(y.values.reshape(-1, 1))
    y_scaled = Y_scaler.transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test, _ = data_transformer.get_train_and_test(
        X_scaled, y_scaled)

    lstm_model = LSTM(input_size=X_train.shape[1], hidden_size=LSTM_HIDDEN_SIZE,
                      num_layers=LSTM_LAYERS, output_size=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)

    early_stopping = EarlyStopping(patience=50, min_delta=0.0001)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        outputs = lstm_model(X_train.unsqueeze(1)).squeeze()
        optimizer.zero_grad()
        train_loss: torch.Tensor = criterion(outputs, y_train)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_outputs = lstm_model(X_test.unsqueeze(1)).squeeze()
            test_loss: torch.Tensor = criterion(test_outputs, y_test)

        test_outputs = Y_scaler.inverse_transform(
            test_outputs.reshape(-1, 1)).flatten()
        y_test_original = Y_scaler.inverse_transform(
            y_test.numpy().reshape(-1, 1)).flatten()

        if (epoch + 1) % 10 == 0:
            r2 = r2_score(y_test_original, test_outputs)
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Test R2: {r2:.4f}")

        early_stopping.update(test_loss, lstm_model)
        if early_stopping.should_stop():
            if r2 > best_combination["r_squared"]:
                logging.info(
                    f"New best combination found: {features}, R2: {r2}")
                best_combination["r_squared"] = r2
                best_combination["features"] = features
            print(f'Early stopping at epoch {epoch}')
            break
