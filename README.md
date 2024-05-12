# Saules enerģijas prognozēšana

Šajā repozitorijā ir kods datu apkopošanai un dažādu modeļu apmācībai maģistra darba ietvaros.

Projekta struktūra ir pieejama faila beigās.

# Solar power forecasting

This repository contains code for gathering data and various models training as a part of masters thesis.

Project structure is available at the end of the file.

** Data scraping **

MongoDb on port 27017 is required as well as Rust installed on the system.

Meteo data scraper is written is Rust programming language and data is saved to MongoDb. The scraper is designed to gather data for the last year for three specific locations (by latitude and longitude). The scraper can be launched as many times as the user wants, but the data will be scraped based on the newest date from the database for a specific object. 

** Model training **
---

```
.
├── config
│   ├── database.py
│   ├── **init**.py
│   └── **pycache**
├── database
│   ├── **init**.py
│   ├── main.py
│   └── **pycache**
├── data_handling
│   ├── **init**.py
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
│   ├── **init**.py
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
│   ├── **init**.py
│   ├── main.py
│   └── **pycache**
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
