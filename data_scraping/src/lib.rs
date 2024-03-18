pub mod logger;
use log::{error, info};
use open_meteo::database::MONGODB_INSTANCE;
use std::{sync::Arc, thread, time};
use tokio::runtime;
use tokio::sync::mpsc::error::TryRecvError;
use tokio::time::Duration;

use lazy_static::lazy_static;
pub mod open_meteo;
use open_meteo::channel::CHANNEL;
use open_meteo::historical_data::RequestHandler as forecast_fetcher;

lazy_static! {
    static ref TOKIO_RT: Arc<runtime::Runtime> = Arc::new(
        runtime::Builder::new_multi_thread()
            .enable_all()
            .worker_threads(4)
            .build()
            .unwrap()
    );
}

pub fn run() {
    logger::init_logger();
    let mut example = forecast_fetcher::new(Arc::clone(&MONGODB_INSTANCE));

    TOKIO_RT.spawn(async move {
        let result = forecast_fetcher::run(&mut example).await;
        if let Err(e) = result {
            error!("Error running weather scraper {:?}", e);
        }
    });

    TOKIO_RT.spawn(async move {
        let receiver_wrapper = CHANNEL.document_receiver();
        let mut receiver = receiver_wrapper.lock().await;
        let mongo = Arc::clone(&MONGODB_INSTANCE);

        loop {
            match receiver.try_recv() {
                Ok(data) => {
                    let mongo_db_guard = mongo.lock().await;
                    // Process the received data here
                    let doc_count = mongo_db_guard.upsert_record(data).await;
                    if let Ok(docs) = doc_count {
                        info!("Upserted: {} docs.", docs.len());
                    } else {
                        error!("Error while updating docs");
                    }
                }
                Err(TryRecvError::Empty) => {
                    continue;
                }
                Err(TryRecvError::Disconnected) => {
                    break;
                }
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    });

    loop {
        thread::sleep(time::Duration::from_millis(10000));
    }
}
