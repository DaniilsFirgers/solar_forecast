use log::info;
use reqwest::Error as ReqErr;
use serde::{Deserialize, Serialize};
use std::error::Error;

const BASE_URL: &str = "https://archive-api.open-meteo.com/v1/archive";
const FEATURES: &str = "&hourly=temperature_2m,relative_humidity_2m,precipitation,rain,surface_pressure,wind_speed_10m,direct_radiation";

pub struct RequestHandler {
    pub latitude: String,
    pub longitude: String,
    pub start_date: String,
    pub end_date: String,
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
    #[serde(rename = "temperature")]
    pub temperature_2m: Vec<f64>,
    #[serde(rename = "relative_humidity")]
    pub relative_humidity_2m: Vec<i64>,
    pub precipitation: Vec<f64>,
    pub rain: Vec<f64>,
    pub surface_pressure: Vec<f64>,
    pub wind_speed_10m: Vec<f64>,
    pub direct_radiation: Vec<f64>,
}

impl RequestHandler {
    pub fn new(latitude: String, longitude: String, start_date: String, end_date: String) -> Self {
        Self {
            latitude,
            longitude,
            start_date,
            end_date,
        }
    }
    pub async fn run(&mut self) -> Result<(), Box<dyn Error>> {
        info!("Historical weather forecast scraper started!");
        self.handle_data_gathering().await?;
        Ok(())
    }

    async fn handle_data_gathering(&mut self) -> Result<(), ReqErr> {
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
        let response: HistoricalWeatherForecast = client.get(&url).send().await?.json().await?;

        println!("Deserialized data: {:?}", response);
        Ok(())
    }
}
