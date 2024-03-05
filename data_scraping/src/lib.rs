pub mod logger;
pub mod open_meteo;
use open_meteo::historical_data::RequestHandler as forecast_fetcher;

pub fn run() {
    logger::init_logger();

    let mut example = open_meteo::historical_data::RequestHandler::new(
        "56.11".to_string(),
        "56.11".to_string(),
        "2024-01-01".to_string(), // Start date
        "2024-12-31".to_string(),
    );
    forecast_fetcher::run(&mut example);
    // std::thread(move || {})
}
