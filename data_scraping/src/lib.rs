pub mod logger;
use log::{error, info};
use open_meteo::database::MongoDb;
use std::f32::consts::E;
use std::{sync::Arc, thread, time};
use tokio::runtime;

use lazy_static::lazy_static;
pub mod open_meteo;
use open_meteo::channel::CHANNEL;
use open_meteo::historical_data::RequestHandler as forecast_fetcher;

lazy_static! {
    static ref TOKIO_RT: Arc<runtime::Runtime> = Arc::new(
        runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(16)
            .build()
            .unwrap()
    );
}

pub fn run() {
    logger::init_logger();

    let mut example = forecast_fetcher::new(
        "52.52".to_string(),
        "13.41".to_string(),
        "2024-02-18".to_string(), // Start date
        "2024-03-03".to_string(),
    );

    TOKIO_RT.spawn(async move {
        let result = forecast_fetcher::run(&mut example).await;
        if let Err(e) = result {
            error!("Error running weather scraper {:?}", e);
        }
    });

    TOKIO_RT.spawn(async move {
        let mongodb_result = MongoDb::new().await;
        let mongodb_res: MongoDb = match mongodb_result {
            Ok(db) => db,
            Err(_) => panic!("Error with mongo db!!"),
        };

        let receiver_wrapper = CHANNEL.document_receiver();
        let mut receiver = receiver_wrapper.lock().await;
        while let Some(data) = receiver.recv().await {
            // Process the received data here
            println!("Received data: {:#?} length", data.len());
            let doc_count = mongodb_res.upsert_record(data).await;
            if let Ok(doc_count) = doc_count {
                info!("Upserted: {} docs", doc_count.len());
            } else {
                error!("Error while updating docs");
            }
        }
    });

    loop {
        thread::sleep(time::Duration::from_millis(10000));
    }
}
