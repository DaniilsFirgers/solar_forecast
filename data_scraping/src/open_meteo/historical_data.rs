use bson::Document;
use log::{error, info};
use mongodb::bson;
use reqwest::Error as ReqErr;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::sync::Mutex;

use super::channel::CHANNEL;
use super::database::MongoDb;
use super::document::FormattedWeatherData;

const BASE_URL: &str = "https://archive-api.open-meteo.com/v1/archive";
const FEATURES: &str = "&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,surface_pressure,wind_speed_10m,direct_radiation";
static NONE_STRING: &str = "None";
pub struct RequestHandler {
    pub latitude: String,
    pub longitude: String,
    pub start_date: String,
    pub end_date: String,
    sender: mpsc::UnboundedSender<Vec<Document>>,
    mongo_db: Arc<Mutex<MongoDb>>,
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
    // #[serde(rename = "temperature")]
    pub temperature_2m: String,
    // #[serde(rename = "relative_humidity")]
    pub relative_humidity_2m: String,
    pub precipitation: String,
    pub rain: String,
    // #[serde(rename = "surface_pressure")]
    pub surface_pressure: String,
    // #[serde(rename = "wind_speed_10m")]
    pub wind_speed_10m: String,
    // #[serde(rename = "direct_radiation")]
    pub direct_radiation: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HourlyData {
    pub time: Vec<String>,
    // #[serde(rename = "temperature")]
    pub temperature_2m: Vec<f64>,
    // #[serde(rename = "relative_humidity")]
    pub relative_humidity_2m: Vec<i64>,
    pub precipitation: Vec<f64>,
    pub rain: Vec<f64>,
    pub surface_pressure: Vec<f64>,
    pub wind_speed_10m: Vec<f64>,
    pub direct_radiation: Vec<f64>,
}

impl RequestHandler {
    pub fn new(
        latitude: String,
        longitude: String,
        start_date: String,
        end_date: String,
        mongo_client: Arc<Mutex<MongoDb>>,
    ) -> Self {
        let sender = CHANNEL.document_sender();

        Self {
            latitude,
            longitude,
            start_date,
            end_date,
            sender,
            mongo_db: mongo_client,
        }
    }
    pub async fn run(&mut self) -> Result<(), Box<dyn Error>> {
        info!("Historical weather forecast scraper started!");
        self.handle_data_gathering().await?;
        Ok(())
    }

    async fn handle_data_gathering(&mut self) -> Result<(), ReqErr> {
        let mongo_guard = &*self.mongo_db.lock().await;
        let res = mongo_guard
            .get_latest_object_doc(self.longitude.clone(), self.latitude.clone())
            .await
            .unwrap();
        println!("res: {:?}", res);

        let url = self.create_url();
        self.get_data(url).await?;
        Ok(())
    }

    fn create_url(&self) -> String {
        let url = format!(
            "{}?latitude={}&longitude={}&start_date={}&end_date={}&{}",
            &BASE_URL, self.latitude, self.longitude, self.start_date, self.end_date, FEATURES
        );
        url
    }
    async fn get_data(&self, url: String) -> Result<(), ReqErr> {
        let client = reqwest::Client::new();
        let mut response: HistoricalWeatherForecast = client.get(&url).send().await?.json().await?;

        self.format_response(&mut response);
        Ok(())
    }
    fn format_response(&self, response: &mut HistoricalWeatherForecast) {
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
            ) = (
                response.hourly.time.get(i),
                response.hourly.temperature_2m.get(i),
                response.hourly.relative_humidity_2m.get(i),
                response.hourly.precipitation.get(i),
                response.hourly.rain.get(i),
                response.hourly.surface_pressure.get(i),
                response.hourly.wind_speed_10m.get(i),
                response.hourly.direct_radiation.get(i),
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
