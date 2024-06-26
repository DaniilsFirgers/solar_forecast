from sklearn.linear_model import Lasso
import pandas as pd

from sklearn.preprocessing import RobustScaler
from config.database import FORECAST_DB_NAME, PRODUCTION_COLLECTION_NAME, WEATHER_COLLECTION_NAME
from database.main import mongo_handler
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from data_handling.transform import DataTransformer, PARAMETERS_NAME_MAP
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler

plt.style.use('ggplot')
matplotlib.use('tkagg')
SPLIT_RATIO = 0.80
OBJECT = "B"
TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
INPUT_FEATURES = ["shortwave_radiation",  'relative_humidity', 'pressure', 'rain',
                  'wind_speed', "temperature", 'direct_normal_irradiance', 'direct_radiation', 'terrestrial_radiation']

SELECTED_FEATURES = ['shortwave_radiation', 'terrestrial_radiation',
                     'relative_humidity', 'pressure']

HYPER_PARAMETERS = {
    "alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "max_iter": [50, 100, 500, 2000, 3000, 4000, 5000]
}

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


def find_best_features(alpha: float = 0.01, max_iter: int = 2000) -> None:
    X = merged_df[INPUT_FEATURES]
    y = merged_df['value']

    # Split the data into training and testing sets
    X_train, X_test, X_val, y_val, y_train, y_test = data_transformer.get_train_and_test_data(
        X, y)

    X_scaler = RobustScaler()
    X_train = X_scaler.fit_transform(X_train)

    y_scaler = RobustScaler()
    y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1))

    lasso_model = Lasso(alpha=alpha, max_iter=max_iter)

    lasso_model.fit(X_train, y_train)

    feature_importance = lasso_model.coef_

    translated_column_names = [PARAMETERS_NAME_MAP.get(
        col, col) for col in INPUT_FEATURES]

    feature_importance_df = pd.DataFrame({
        "Feature": translated_column_names,
        "Importance": feature_importance
    })

    feature_importance_df = feature_importance_df.reindex(
        feature_importance_df['Importance'].sort_values(ascending=True).index)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'],
             feature_importance_df['Importance'], color='green')
    plt.xlabel('Svarīgums')
    plt.title('Lasso modeļa parametru svarīgums')
    # plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_selection/lasso_feature_importance.png')
    plt.show()


def find_best_hyperparameters() -> tuple[float, int]:
    X = merged_df[SELECTED_FEATURES]
    y = merged_df['value']

    X_scaler = RobustScaler()
    X_scaled = X_scaler.fit_transform(X)

    y_scaler = RobustScaler()
    y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=SPLIT_RATIO, random_state=42)
    lass_model = Lasso()
    random_search = GridSearchCV(
        estimator=lass_model, param_grid=HYPER_PARAMETERS, cv=5,  scoring="r2")

    random_search.fit(X_train, y_train)
    # Get the best parameters and best score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    random_search.fit(X_test, y_test)
    best_val_score = random_search.best_score_

    best_alpha = best_params.get("alpha")
    best_max_iters = best_params.get("max_iter")
    return best_alpha, best_max_iters


if __name__ == "__main__":
    best_alpha, best_max_iters = find_best_hyperparameters()
    print("Best alpha:", best_alpha)
    print("Best max_iter:", best_max_iters)
    best_alpha, best_max_iters = find_best_features(best_alpha, best_max_iters)
