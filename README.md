# Saules enerģijas prognozēšana

Šajā repozitorijā ir kods datu apkopošanai un dažādu modeļu apmācībai maģistra darba ietvaros.

Projekta struktūra ir pieejama faila beigās.

# Solar power forecasting

This repository contains code for gathering data and various models training as a part of masters thesis.

Project structure is available at the end of the file.

**Data scraping**

Meteo data scraper is written is Rust programming language and data is saved to MongoDb. The scraper is designed to gather data for the last year for three specific locations (by latitude and longitude). The scraper can be launched as many times as the user wants, but the data will be scraped based on the newest date from the database for a specific object.

- Scraper can be launched in docker by running `docker compose up` in /data_scraping (can take up a considerabel amount of disk space);
- Or `MONGO_URL` environment variable can be added to run on the default port with `mongodb://localhost:27017`. Then cd /data_scraping and launch script with `cargo run`. You need rust and mongodb installed on your local machine;

**Model training**

- `main.py` script has code for training, validation and testing of machine learning models;
- If retraining for regressions is needed set `LR_NEED_TRAINING` to true, if retraining of neural networks is needed set `NN_NEED_TRAINING` to true. This script also saves loss and results plots to /plots folder. Best model weights are saved in the /trained_models folder;
- /feature_selection folder has scripts for finding out best parameters for each model (as well as previous runs best parameters saved as plots);
- /database folder has a class and associated functions for database connections and records insertion;
- In /data_handling folder transform.py file includes classes for models early stopping, multiple plots creation and data transformation as well as a few utility functions;
- In /config folder in database.py file there are database related connection variables and URL's;
- If the used decides to launch data scraper locally without docker then `MONGO_URL` should be set to `mongodb://localhost:27017`, otherwise to `mongodb://localhost:8001`
- `vizualize.ipynb` contains various scripts for data vizualizations such as historical weather data or correlation matrix;

The user has two options, either to download required packages from `requirements.txt` to local machine, or run in a virtual environment. Create it with `python3 -m venv myenv` and run it with `source activate`.
Deactivate virtual environment with the command `source deactivate`

```
.
├── config
│   ├── database.py
│   ├── init.py
│   └── pycache
├── database
│   ├── init.py
│   ├── main.py
│   └── pycache
├── data_handling
│   ├── init.py
│   └── transform.py
├── data_scraping
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── src
│      ├── lib.rs
│      ├── logger.rs
│      ├── main.rs
│      └── open_meteo
│      ├── channel.rs
│      ├── database.rs
│      ├── document.rs
│      ├── historical_data.rs
│      └── mod.rs
│  
├── feature_selection
│   ├── correlation_A.png
│   ├── init.py
│   ├── lasso_feature_importance.png
│   ├── lasso.py
│   ├── linear_regression.py
│   ├── linear_regression_rfe_feature_rankings.png
│   ├── lstm_features.log
│   ├── lstm.py
│   ├── rnn_features.log
│   └── rnn.py
├── main.py
├── models
│   ├── init.py
│   ├── main.py
│   └── pycache
│  
│  
├── plots
├── project_tree.txt
├── README.md
├── requirements.txt
├── trained_models
   ├── best_GRU_weights_A.pt
   ├── best_GRU_weights_B.pt
   ├── best_LSTM_weights_A.pt
   ├── best_LSTM_weights_B.pt
   ├── best_RNN_weights_A.pt
   ├── best_RNN_weights_B.pt
   ├── gb_A.pkl
   ├── gb_B.pkl
   ├── lasso_A.pkl
   ├── lasso_B.pkl
   ├── linear_regression_A.pkl
   └── linear_regression_B.pkl
```
