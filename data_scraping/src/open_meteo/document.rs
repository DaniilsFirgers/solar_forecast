use std::{error::Error, fmt};

use bson::{doc, Document};
use mongodb::bson;

#[derive(Debug, Clone)]
pub struct FormattedWeatherData {
    pub datetime: String,
    pub temperature: f64,
    pub relative_humidity: i64,
    pub precipitation: f64,
    pub rain: f64,
    pub pressure: f64,
    pub wind_speed: f64,
    pub direct_radiation: f64,
}

impl FormattedWeatherData {
    pub fn new(
        datetime: String,
        temperature: f64,
        relative_humidity: i64,
        precipitation: f64,
        rain: f64,
        pressure: f64,
        wind_speed: f64,
        direct_radiation: f64,
    ) -> Self {
        Self {
            datetime,
            temperature,
            relative_humidity,
            precipitation,
            rain,
            pressure,
            wind_speed,
            direct_radiation,
        }
    }
}

impl Into<Document> for FormattedWeatherData {
    fn into(self) -> Document {
        doc! {
            "datetime": self.datetime,
            "temperature": self.temperature,
            "relative_humidity": self.relative_humidity,
            "precipitation": self.precipitation,
            "rain": self.rain,
            "pressure": self.pressure,
            "wind_speed": self.wind_speed,
            "direct_radiation": self.direct_radiation,
        }
    }
}

#[derive(Debug)]
pub enum ConversionError {
    MissingField(String),
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConversionError::MissingField(field) => {
                write!(f, "Field '{}' missing or invalid type", field)
            }
        }
    }
}
impl Error for ConversionError {}

impl TryInto<FormattedWeatherData> for Document {
    type Error = ConversionError;

    fn try_into(self) -> Result<FormattedWeatherData, Self::Error> {
        let datetime = match self.get_str("datetime") {
            Ok(datetime) => datetime.to_string(),
            Err(_) => return Err(ConversionError::MissingField("datetime".to_string())),
        };

        let temperature = match self.get_f64("temperature") {
            Ok(temperature) => temperature,
            Err(_) => return Err(ConversionError::MissingField("temperature".to_string())),
        };

        let relative_humidity = match self.get_i64("relative_humidity") {
            Ok(relative_humidity) => relative_humidity,
            Err(_) => {
                return Err(ConversionError::MissingField(
                    "relative_humidity".to_string(),
                ))
            }
        };

        let precipitation = match self.get_f64("precipitation") {
            Ok(precipitation) => precipitation,
            Err(_) => return Err(ConversionError::MissingField("precipitation".to_string())),
        };

        let rain = match self.get_f64("rain") {
            Ok(rain) => rain,
            Err(_) => return Err(ConversionError::MissingField("rain".to_string())),
        };

        let pressure = match self.get_f64("pressure") {
            Ok(pressure) => pressure,
            Err(_) => return Err(ConversionError::MissingField("pressure".to_string())),
        };

        let wind_speed = match self.get_f64("wind_speed") {
            Ok(wind_speed) => wind_speed,
            Err(_) => return Err(ConversionError::MissingField("wind_speed".to_string())),
        };

        let direct_radiation = match self.get_f64("direct_radiation") {
            Ok(direct_radiation) => direct_radiation,
            Err(_) => {
                return Err(ConversionError::MissingField(
                    "direct_radiation".to_string(),
                ))
            }
        };

        Ok(FormattedWeatherData {
            datetime,
            temperature,
            relative_humidity,
            precipitation,
            rain,
            pressure,
            wind_speed,
            direct_radiation,
        })
    }
}
