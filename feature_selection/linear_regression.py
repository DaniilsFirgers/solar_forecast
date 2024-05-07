import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from config.database import FORECAST_DB_NAME, PRODUCTION_COLLECTION_NAME, WEATHER_COLLECTION_NAME
from database.main import mongo_handler
import matplotlib.pyplot as plt
import matplotlib
from data_handling.transform import DataTransformer, PARAMETERS_NAME_MAP
from sklearn.preprocessing import RobustScaler

plt.style.use('ggplot')
matplotlib.use('tkagg')
OBJECT = "B"
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
INPUT_FEATURES = ["shortwave_radiation",  'relative_humidity', 'pressure', 'rain',
                  'wind_speed', "temperature", 'direct_normal_irradiance', 'direct_radiation', 'terrestrial_radiation']
filter_query = {
    "object_name": OBJECT
}

historical_data = mongo_handler.retrieve_production_data(
    FORECAST_DB_NAME, PRODUCTION_COLLECTION_NAME, filter_query)
weather_data = mongo_handler.retrieve_weather_data(
    FORECAST_DB_NAME, WEATHER_COLLECTION_NAME, filter_query)
if historical_data is None or weather_data is None:
    print("Error retrieving data from MongoDB.")
    exit(1)

data_transformer = DataTransformer(historical_data, weather_data, test_ratio=TEST_SPLIT,
                                   valiation_ratio=VALIDATION_SPLIT, train_ratio=TRAIN_SPLIT)

merged_df, index = data_transformer.get_merged_df()


def find_best_features():
    X = merged_df[INPUT_FEATURES]
    y = merged_df['value']

    lr = LinearRegression()

    # Split the data into training and testing sets
    X_train, X_test, X_val, y_val, y_train, y_test = data_transformer.get_train_and_test_data(
        X, y)

    X_scaler = RobustScaler()
    X_train = X_scaler.fit_transform(X_train)

    y_scaler = RobustScaler()
    y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

    selector = RFE(estimator=lr, n_features_to_select=5)

    selector.fit(X_train, y_train)
    selected_features_indices = selector.support_
    selected_features = X.columns[selected_features_indices]

    feature_importance = selector.ranking_
    selected_features = selected_features.to_list()

    feature_importance_df = pd.DataFrame(
        {'Feature': INPUT_FEATURES, 'Ranking': feature_importance})

    # Sort features by their ranking
    feature_importance_df.sort_values(by='Ranking', inplace=True)

    translated_column_names = [PARAMETERS_NAME_MAP.get(
        col, col) for col in feature_importance_df['Feature']]
    plt.figure(figsize=(10, 6))
    plt.barh(translated_column_names,
             feature_importance_df['Ranking'], color='green')
    plt.xlabel('Svarīgums')
    plt.title('RFE Linearas regresijas modeļa parametru svarīgums')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig('feature_selection/linear_regression_rfe_feature_rankings.png')
    plt.show()


if __name__ == "__main__":
    find_best_features()
