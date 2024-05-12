# Saules enerģijas prognozēšana

Šajā repozitorijā ir kods datu apkopošanai un dažādu modeļu apmācībai maģistra darba ietvaros.

Projekta struktūra ir pieejama faila beigās.

# Solar power forecasting

This repository contains code for gathering data and various models training as a part of masters thesis.

Project structure is available at the end of the file.

---

```
.
├── config
│   ├── database.py
│   ├── **init**.py
│   └── **pycache**
│   ├── database.cpython-310.pyc
│   └── **init**.cpython-310.pyc
├── database
│   ├── **init**.py
│   ├── main.py
│   └── **pycache**
│   ├── **init**.cpython-310.pyc
│   └── main.cpython-310.pyc
├── data_handling
│   ├── **init**.py
│   ├── **pycache**
│   │   ├── **init**.cpython-310.pyc
│   │   └── transform.cpython-310.pyc
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
│   ├── **init**.cpython-310.pyc
│   └── main.cpython-310.pyc
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
