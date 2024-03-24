use std::{error::Error, fmt};

use bson::{doc, Document};
use mongodb::bson;

#[derive(Debug, Clone)]
pub struct FormattedWeatherData {
    pub start_time: String,
    pub temperature: f64,
    pub relative_humidity: i64,
    pub precipitation: f64,
    pub rain: f64,
    pub pressure: f64,
    pub wind_speed: f64,
    pub direct_radiation: f64,
    pub object_name: String,
    pub direct_radiation_instant: f64,
    pub direct_normal_irradiance: f64,
    pub direct_normal_irradiance_instant: f64,
    pub shortwave_radiation: f64,
    pub diffuse_radiation: f64,
    pub terrestrial_radiation: f64,
    pub terrestrial_radiation_instant: f64,
}

impl FormattedWeatherData {
    pub fn new(
        start_time: String,
        temperature: f64,
        relative_humidity: i64,
        precipitation: f64,
        rain: f64,
        pressure: f64,
        wind_speed: f64,
        direct_radiation: f64,
        object_name: String,
        direct_radiation_instant: f64,
        direct_normal_irradiance: f64,
        direct_normal_irradiance_instant: f64,
        shortwave_radiation: f64,
        diffuse_radiation: f64,
        terrestrial_radiation: f64,
        terrestrial_radiation_instant: f64,
    ) -> Self {
        Self {
            start_time,
            temperature,
            relative_humidity,
            precipitation,
            rain,
            pressure,
            wind_speed,
            direct_radiation,
            object_name,
            direct_radiation_instant,
            direct_normal_irradiance,
            direct_normal_irradiance_instant,
            shortwave_radiation,
            diffuse_radiation,
            terrestrial_radiation,
            terrestrial_radiation_instant,
        }
    }
}

impl Into<Document> for FormattedWeatherData {
    fn into(self) -> Document {
        doc! {
            "start_time": self.start_time,
            "temperature": self.temperature,
            "relative_humidity": self.relative_humidity,
            "precipitation": self.precipitation,
            "rain": self.rain,
            "pressure": self.pressure,
            "wind_speed": self.wind_speed,
            "direct_radiation": self.direct_radiation,
            "object_name": self.object_name,
            "direct_radiation_instant": self.direct_radiation_instant,
            "direct_normal_irradiance": self.direct_normal_irradiance,
            "direct_normal_irradiance_instant": self.direct_normal_irradiance_instant,
            "shortwave_radiation": self.shortwave_radiation,
            "diffuse_radiation": self.diffuse_radiation,
            "terrestrial_radiation": self.terrestrial_radiation,
            "terrestrial_radiation_instant": self.terrestrial_radiation_instant,
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
        let start_time = match self.get_str("start_time") {
            Ok(start_time) => start_time.to_string(),
            Err(_) => return Err(ConversionError::MissingField("start_time".to_string())),
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
        let object_name: String = match self.get_str("object_name") {
            Ok(object_name) => object_name.into(),
            Err(_) => return Err(ConversionError::MissingField("lat".to_string())),
        };

        let direct_radiation_instant = match self.get_f64("direct_radiation_instant") {
            Ok(direct_radiation_instant) => direct_radiation_instant,
            Err(_) => {
                return Err(ConversionError::MissingField(
                    "direct_radiation_instant".to_string(),
                ))
            }
        };
        let direct_normal_irradiance = match self.get_f64("direct_normal_irradiance") {
            Ok(direct_normal_irradiance) => direct_normal_irradiance,
            Err(_) => {
                return Err(ConversionError::MissingField(
                    "direct_normal_irradiance".to_string(),
                ))
            }
        };

        let direct_normal_irradiance_instant =
            match self.get_f64("direct_normal_irradiance_instant") {
                Ok(direct_normal_irradiance_instant) => direct_normal_irradiance_instant,
                Err(_) => {
                    return Err(ConversionError::MissingField(
                        "direct_normal_irradiance_instant".to_string(),
                    ))
                }
            };

        let shortwave_radiation = match self.get_f64("shortwave_radiation") {
            Ok(shortwave_radiation) => shortwave_radiation,
            Err(_) => {
                return Err(ConversionError::MissingField(
                    "shortwave_radiation".to_string(),
                ))
            }
        };

        let diffuse_radiation = match self.get_f64("diffuse_radiation") {
            Ok(diffuse_radiation) => diffuse_radiation,
            Err(_) => {
                return Err(ConversionError::MissingField(
                    "diffuse_radiation".to_string(),
                ))
            }
        };

        let terrestrial_radiation = match self.get_f64("terrestrial_radiation") {
            Ok(terrestrial_radiation) => terrestrial_radiation,
            Err(_) => {
                return Err(ConversionError::MissingField(
                    "terrestrial_radiation".to_string(),
                ))
            }
        };

        let terrestrial_radiation_instant = match self.get_f64("terrestrial_radiation_instant") {
            Ok(terrestrial_radiation_instant) => terrestrial_radiation_instant,
            Err(_) => {
                return Err(ConversionError::MissingField(
                    "terrestrial_radiation_instant".to_string(),
                ))
            }
        };

        Ok(FormattedWeatherData {
            start_time,
            temperature,
            relative_humidity,
            precipitation,
            rain,
            pressure,
            wind_speed,
            direct_radiation,
            object_name,
            direct_radiation_instant,
            direct_normal_irradiance,
            direct_normal_irradiance_instant,
            shortwave_radiation,
            diffuse_radiation,
            terrestrial_radiation,
            terrestrial_radiation_instant,
        })
    }
}
