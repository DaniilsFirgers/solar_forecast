import torch
import torch.nn as nn
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from config.database import PRODUCTION_HISTORY_DB_NAME, PRODUCTION_HISTORY_COLLECTION_NAME, PRODUCTION_WEATHER_COLLECTION_NAME
from database.main import mongo_handler
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from data_handling.transform import EarlyStopping
from models.main import LSTM

plt.style.use('ggplot')
matplotlib.use('tkagg')
SPLIT_RATIO = 0.8
BATCH_SIZE = 32
NUM_EPOCHS = 1000
LSTM_HIDDEN_SIZE = 64
LSTM_LAYERS = 2
window = 1

filter_query = {
    "object_name": "A"
}

historical_data = mongo_handler.retrieve_production_data(
    PRODUCTION_HISTORY_DB_NAME, PRODUCTION_HISTORY_COLLECTION_NAME, filter_query)
weather_data = mongo_handler.retrieve_weather_data(
    PRODUCTION_HISTORY_DB_NAME, PRODUCTION_WEATHER_COLLECTION_NAME, filter_query)
if historical_data is None or weather_data is None:
    print("Error retrieving data from MongoDB.")
    exit(1)

historical_data['start_time'] = pd.to_datetime(historical_data['start_time'])
weather_data['start_time'] = pd.to_datetime(weather_data['start_time'])

merged_df = pd.merge(historical_data, weather_data,
                     on='start_time', how='inner')

scaler = MinMaxScaler()
merged_df.set_index("start_time", inplace=True)

X = merged_df[['direct_radiation', 'precipitation', 'pressure',
               'rain', 'relative_humidity', 'temperature', 'wind_speed']]
y = merged_df['value']

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2,  random_state=42)

X_train = torch.from_numpy(X_train).float()
y_train = torch.squeeze(torch.from_numpy(y_train).float())

X_test = torch.from_numpy(X_test).float()
y_test = torch.squeeze(torch.from_numpy(y_test).float())
lstm_model = LSTM(input_size=X_train.shape[1], hidden_size=LSTM_HIDDEN_SIZE,
                  num_layers=LSTM_LAYERS, output_size=1)

criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)


train_losses = []
test_losses = []


early_stopping = EarlyStopping(patience=50, min_delta=0.0001)

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

    # val_loss = running_vloss / len(test_loader)
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

    test_losses.append(test_loss)
    early_stopping.update(test_loss, lstm_model)
    if early_stopping.should_stop():
        print(f'Early stopping at epoch {epoch}')
        break

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(train_losses) + 1), test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for LSTM Model')
plt.legend()
plt.show()
