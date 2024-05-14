# Saules enerģijas prognozēšana

Šajā repozitorijā ir kods datu apkopošanai un dažādu modeļu apmācībai maģistra darba ietvaros.

Projekta struktūra ir pieejama faila beigās.

**Datu iegūšana**

Meteo datu savācējs ir rakstīts Rust programmēšanas valodā, un dati tiek saglabāti MongoDb. Skreperis ir paredzēts, lai apkopotu datus par pēdējo gadu trīs konkrētās vietās (pēc platuma un garuma). Skreperi var palaist tik reižu, cik lietotājs vēlas, bet dati tiks saskrāpēti, pamatojoties uz jaunāko datumu no datubāzes konkrētam objektam.

- Datu vācēju var palaist docker vidē, palaižot ar `docker compose up` /data_scraping (var aizņemt ievērojamu diska vietas daudzumu);
- Vai arī var pievienot `MONGO_URL` vides mainīgo, lai palaistu noklusējuma portā ar `mongodb://localhost:27017`. Pēc tam direktoriju nomainiet uz /data_scraping un palaižiet skriptu ar `cargo run`. Vietējā datorā ir jābūt instalētam rust un mongodb;

**Modeļu apmācība**

`main.py` skripts satur mašīnmācīšanās modeļu apmācības, validācijas un testēšanas kodu;

- Ja nepieciešams pārmācīt regresiju, iestatiet `LR_NEED_TRAINING` uz true, ja nepieciešams pārmācīt neironu tīklus, iestatiet `NN_NEED_TRAINING` uz true. Šis skripts arī saglabā zaudējumu un rezultātu diagrammas mapē /plots. Labākie modeļu svari tiek saglabāti mapē /trained_models;
- mapē /feature_selection ir skripti, lai noskaidrotu katra modeļa labākos parametrus (kā arī iepriekšējo darbību labākos parametrus, kas saglabāti kā grafiki);
- mapē /database ir klase un saistītās funkcijas datu bāzes savienojumiem un ierakstu ievietošanai;
- mapē /data_handling mapē transform.py failā ir klases modeļu agrīnai apstāšanai, vairāku grafiku izveidei un datu pārveidošanai, kā arī dažas palīgfunkcijas;
- mapē /config database.py failā ir ar datubāzi saistītie savienojuma mainīgie un URL;
- Ja lietotājs nolemj palaist datu skrāpi lokāli bez docker, tad `MONGO_URL` jāiestata uz `mongodb://localhost:27017`, citādi uz `mongodb://localhost:8001`.
- `vizualize.ipynb` satur dažādus skriptus datu vizualizācijai, piemēram, vēsturisko laikapstākļu datu vai korelācijas matricas attēlošanai;

Lietotājam ir divas iespējas - vai nu lejupielādēt nepieciešamās paketes no `requirements.txt` uz vietējo mašīnu, vai arī palaist virtuālajā vidē. Izveidojiet to ar `python3 -m venv myenv` un palaidiet to ar `source activate`.
Deaktivējiet virtuālo vidi ar komandu `source deactivate`.

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
