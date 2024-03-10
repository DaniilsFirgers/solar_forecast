use bson::{doc, Document};
use mongodb::bson;

#[derive(Debug, Clone)]
pub struct FormattedWeatherData<'a> {
    pub datetime: &'a String,
    pub temperature: &'a f64,
    pub relative_humidity: &'a i64,
    pub precipitation: &'a f64,
    pub rain: &'a f64,
    pub pressure: &'a f64,
    pub wind_speed: &'a f64,
    pub direct_radiation: &'a f64,
}

impl<'a> FormattedWeatherData<'a> {
    pub fn new(
        datetime: &'a String,
        temperature: &'a f64,
        relative_humidity: &'a i64,
        precipitation: &'a f64,
        rain: &'a f64,
        pressure: &'a f64,
        wind_speed: &'a f64,
        direct_radiation: &'a f64,
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

impl<'a> Into<Document> for FormattedWeatherData<'a> {
    fn into(self) -> Document {
        doc! {
            "datetime": self.datetime,
            "temperature": self.temperature,
            "relative_humidity": self.relative_humidity,
            "precipitation": self.precipitation,
            "rain": self.rain,
            "pressure": self.pressure,
            "wind_speed": *self.wind_speed,
            "direct_radiation": self.direct_radiation,
        }
    }
}
