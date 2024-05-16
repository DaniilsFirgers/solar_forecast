use bson::Document;
use chrono::DateTime;
use chrono::Duration;
use chrono::NaiveDateTime;
use chrono::ParseError;
use chrono::Timelike;
use chrono::Utc;
use log::{error, info};
use mongodb::bson;
use reqwest::Error as ReqErr;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;

use super::channel::CHANNEL;
use super::database::MongoDb;
use super::document::FormattedWeatherData;

const BASE_URL: &str = "https://archive-api.open-meteo.com/v1/archive";

#[derive(Debug, Clone)]
struct Coordinates {
    lon: String,
    lat: String,
}

#[derive(Clone)]
pub struct RequestHandler {
    sender: mpsc::UnboundedSender<Vec<Document>>,
    mongo_db: Arc<Mutex<MongoDb>>,
    locations: HashMap<String, Coordinates>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HistoricalWeatherForecast {
    pub latitude: f64,
    pub longitude: f64,
    // #[serde(rename = "generationtime_ms")]
    pub generationtime_ms: f64,
    // #[serde(rename = "utc_offset_seconds")]
    pub utc_offset_seconds: i64,
    pub timezone: String,
    pub timezone_abbreviation: String,
    pub elevation: f64,
    pub hourly_units: HourlyUnits,
    // #[serde(rename = "hourly_data")]
    pub hourly: HourlyData,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HourlyUnits {
    pub time: String,
    pub temperature_2m: String,
    pub relative_humidity_2m: String,
    pub precipitation: String,
    pub rain: String,
    pub surface_pressure: String,
    pub wind_speed_10m: String,
    pub direct_radiation: String,
    pub direct_radiation_instant: String,
    pub direct_normal_irradiance: String,
    pub direct_normal_irradiance_instant: String,
    pub shortwave_radiation: String,
    pub diffuse_radiation: String,
    pub terrestrial_radiation: String,
    pub terrestrial_radiation_instant: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HourlyData {
    pub time: Vec<String>,
    pub temperature_2m: Vec<f64>,
    pub relative_humidity_2m: Vec<i64>,
    pub precipitation: Vec<f64>,
    pub rain: Vec<f64>,
    pub surface_pressure: Vec<f64>,
    pub wind_speed_10m: Vec<f64>,
    pub direct_radiation: Vec<f64>,
    pub direct_radiation_instant: Vec<f64>,
    pub direct_normal_irradiance: Vec<f64>,
    pub direct_normal_irradiance_instant: Vec<f64>,
    pub shortwave_radiation: Vec<f64>,
    pub diffuse_radiation: Vec<f64>,
    pub terrestrial_radiation: Vec<f64>,
    pub terrestrial_radiation_instant: Vec<f64>,
}

impl RequestHandler {
    pub fn new(mongo_client: Arc<Mutex<MongoDb>>) -> Self {
        let sender = CHANNEL.document_sender();

        let mut locations: HashMap<String, Coordinates> = HashMap::new();
        locations.insert(
            String::from("A"),
            Coordinates {
                lat: String::from("56.722988"),
                lon: String::from("21.575005"),
            },
        );

        locations.insert(
            String::from("B"),
            Coordinates {
                lat: String::from("57.216143"),
                lon: String::from("22.523923"),
            },
        );

        Self {
            sender,
            mongo_db: mongo_client,
            locations,
        }
    }
    pub async fn run(&mut self) -> Result<(), Box<dyn Error>> {
        info!("Historical weather forecast scraper started!");

        let mut locations = self.locations.clone();
        for location in &mut locations {
            info!("Scraping historical weather data for {}", location.0);
            match self.handle_data_gathering(&location).await {
                Ok(_) => (),
                Err(e) => {
                    error!(
                        "Error while data gathering for object {} - {}",
                        location.0, e
                    );
                }
            };
        }

        Ok(())
    }

    fn get_end_date(&self) -> DateTime<Utc> {
        let mut utc_now: DateTime<Utc> = Utc::now();
        utc_now = utc_now.with_minute(0).unwrap().with_second(0).unwrap();
        let two_days_ago = utc_now - Duration::try_days(2).unwrap();

        two_days_ago
    }

    fn parse_datetime_to_utc(&self, datetime: String) -> Result<DateTime<Utc>, ParseError> {
        println!("{}", datetime);
        let parsed_start_time = match NaiveDateTime::parse_from_str(&datetime, "%Y-%m-%dT%H:%M") {
            Ok(dt) => dt,
            Err(e) => return Err(e),
        };

        let utc_start_time = parsed_start_time.and_utc();
        Ok(utc_start_time)
    }

    async fn handle_data_gathering(
        &mut self,
        coordinates: &(&String, &mut Coordinates),
    ) -> Result<(), Box<dyn Error>> {
        let end_date = self.get_end_date();
        // TODO: use a default datetiem string here
        let mut start_date = self.parse_datetime_to_utc(String::from("2023-01-01T00:00"))?;
        let mongo_db = Arc::clone(&self.mongo_db);
        let mongo_guard = mongo_db.lock().await;
        if let Ok(res) = mongo_guard.get_latest_object_doc(&coordinates.0).await {
            start_date = self.parse_datetime_to_utc(res.start_time)?;
        } else {
            println!("No document found, starting from {}", start_date);
        }

        while start_date < end_date {
            let intermmediate_end = start_date + Duration::try_weeks(1).unwrap();
            let url = self.create_url(start_date, intermmediate_end, coordinates.1);
            self.get_data(url, &coordinates).await?;

            // Add one week to start_date
            start_date = intermmediate_end;
        }

        Ok(())
    }
    fn get_parameters_url_postfix(&self) -> String {
        let parameters_list = vec![
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "surface_pressure",
            "wind_speed_10m",
            "direct_radiation",
            "direct_radiation_instant",
            "direct_normal_irradiance",
            "direct_normal_irradiance_instant",
            "shortwave_radiation",
            "diffuse_radiation",
            "terrestrial_radiation",
            "terrestrial_radiation_instant",
        ];
        let mut parameters = String::new();

        for (index, parameter) in parameters_list.iter().enumerate() {
            parameters.push_str(parameter);
            if index < parameters_list.len() - 1 {
                parameters.push_str(",");
            }
        }
        parameters
    }

    fn create_url(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        coordinates: &Coordinates,
    ) -> String {
        let start_date = start.format("%Y-%m-%d").to_string();
        let end_date = end.format("%Y-%m-%d").to_string();

        let params_postfix = self.get_parameters_url_postfix();

        let url = format!(
            "{}?latitude={}&longitude={}&start_date={}&end_date={}&hourly={}",
            &BASE_URL, coordinates.lat, coordinates.lon, start_date, end_date, params_postfix
        );
        println!("{}", url);
        url
    }
    async fn get_data(
        &self,
        url: String,
        coordinates: &(&String, &mut Coordinates),
    ) -> Result<(), ReqErr> {
        let client = reqwest::Client::new();
        let mut response: HistoricalWeatherForecast = client.get(&url).send().await?.json().await?;

        self.format_response(&mut response, coordinates);
        Ok(())
    }
    fn format_response(
        &self,
        response: &mut HistoricalWeatherForecast,
        coordinates: &(&String, &mut Coordinates),
    ) {
        let data_len = response.hourly.time.len();
        let mut data_vec: Vec<Document> = Vec::with_capacity(data_len);
        let none_string = String::from("None");
        let start = response.hourly.time.get(0).unwrap_or(&none_string);
        let end = response
            .hourly
            .time
            .get(data_len - 1)
            .unwrap_or(&none_string);
        info!(
            "Scraping weather data for the period from: {} to {}",
            start, end
        );

        for i in 0..data_len {
            if let (
                Some(datetime),
                Some(temperature),
                Some(relative_humidity),
                Some(precipitation),
                Some(rain),
                Some(surface_pressure),
                Some(wind_speed),
                Some(direct_radiation),
                Some(direct_radiation_instant),
                Some(direct_normal_irradiance),
                Some(direct_normal_irradiance_instant),
                Some(shortwave_radiation),
                Some(diffuse_radiation),
                Some(terrestrial_radiation),
                Some(terrestrial_radiation_instant),
            ) = (
                response.hourly.time.get(i),
                response.hourly.temperature_2m.get(i),
                response.hourly.relative_humidity_2m.get(i),
                response.hourly.precipitation.get(i),
                response.hourly.rain.get(i),
                response.hourly.surface_pressure.get(i),
                response.hourly.wind_speed_10m.get(i),
                response.hourly.direct_radiation.get(i),
                response.hourly.direct_radiation_instant.get(i),
                response.hourly.direct_normal_irradiance.get(i),
                response.hourly.direct_normal_irradiance_instant.get(i),
                response.hourly.shortwave_radiation.get(i),
                response.hourly.diffuse_radiation.get(i),
                response.hourly.terrestrial_radiation.get(i),
                response.hourly.terrestrial_radiation_instant.get(i),
            ) {
                let formatted_data = FormattedWeatherData::new(
                    datetime.to_string(),
                    *temperature,
                    *relative_humidity,
                    *precipitation,
                    *rain,
                    *surface_pressure,
                    *wind_speed,
                    *direct_radiation,
                    coordinates.0.clone(),
                    *direct_radiation_instant,
                    *direct_normal_irradiance,
                    *direct_normal_irradiance_instant,
                    *shortwave_radiation,
                    *diffuse_radiation,
                    *terrestrial_radiation,
                    *terrestrial_radiation_instant,
                );
                let mongo_doc: Document = formatted_data.into();
                data_vec.push(mongo_doc);
            } else {
                error!("Missing data for index {i}");
            }
        }
        if let Err(e) = self.sender.send(data_vec.clone()) {
            error!("Could not send mongo documents to receiver: {e}");
        };
    }
}
