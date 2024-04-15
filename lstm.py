import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from config.database import FORECAST_DB_NAME, PRODUCTION_COLLECTION_NAME, WEATHER_COLLECTION_NAME
from database.main import mongo_handler
import matplotlib.pyplot as plt
import matplotlib
from data_handling.transform import EarlyStopping, PlotLoss, DataTransformer, PlotPredictions
from models.main import LSTM
from sklearn.metrics import r2_score
from matplotlib.dates import AutoDateLocator, DateFormatter

plt.style.use('ggplot')
matplotlib.use('tkagg')
SPLIT_RATIO = 0.75
NUM_EPOCHS = 1000
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
OBJECTS = ['C']
INPUT_FEATURES = ['direct_radiation', 'pressure', 'relative_humidity',
                  'temperature', 'terrestrial_radiation', 'wind_speed', 'month', 'day_of_week', 'hour', 'value_lag_1']
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

    model_plot_title = f'LSTM modeļa apmācības un validācijas zaudējumi {object} objektam'
    model_save_path = f'plots/LSTM-{object}-loss.png'

    results_plot_title = f'Ražošanas prognozes pret patiesajām vērtībām LSTM modelim {object} objektam'
    results_save_path = f'plots/LSTM-{object}-results.png'

    data_transformer = DataTransformer(historical_data, weather_data)

    merged_df = data_transformer.get_merged_df()
    data_transformer.add_lagged_features(LAGGED_FEATURES, LAG_STEPS)
    X = merged_df[INPUT_FEATURES]
    y = merged_df['value']

    X_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)

    Y_scaler = MinMaxScaler()
    Y_scaler.fit(y.values.reshape(-1, 1))
    y_scaled = Y_scaler.transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test, ground_truth_df = data_transformer.get_train_and_test(
        X_scaled, y_scaled)

    lstm_model = LSTM(input_size=X_train.shape[1], hidden_size=LSTM_HIDDEN_SIZE,
                      num_layers=LSTM_LAYERS, output_size=1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    train_losses = []
    test_losses = []

    predicted = []
    ground_truth = []

    early_stopping = EarlyStopping(patience=40, min_delta=0.0001)

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

        test_losses.append(test_loss.detach().numpy())
        train_losses.append(train_loss.detach().numpy())

        early_stopping.update(test_loss, lstm_model)
        if early_stopping.should_stop():
            predicted = test_outputs
            ground_truth = y_test_original
            print(f'Early stopping at epoch {epoch}')
            break
    model_plot = PlotLoss('LSTM', object_name=object, title=model_plot_title,
                          save_path=model_save_path, x_data=train_losses, y_data=test_losses)
    results_plot = PlotPredictions('LSTM', object_name=object, title=results_plot_title, save_path=results_save_path, ground_truth=ground_truth_df,
                                   x_data=ground_truth, y_data=predicted)
    model_plot.create_plot()

    index_array = ground_truth_df.index.to_numpy()
    plt.plot(index_array, predicted,
             color='blue', label='Prognozētās vērtības')
    # Plot truth values as a red line
    plt.plot(index_array, ground_truth,
             color='green', label='Patiesās vērtības')
    plt.xlabel('Datums')
    plt.ylabel('Saražotā saules enerģija, Kwh')
    plt.title(f'Ražošanas prognozes un patiesās vērtības {object} objektam')
    plt.legend()
    plt.grid(True)

    plt.gca().xaxis.set_major_locator(AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter('%d-%m-%y %H:%M'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    # plt.savefig(results_save_path)
    plt.close()
