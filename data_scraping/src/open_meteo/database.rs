use mongodb::{
    bson::{doc, Bson, Document},
    error::Error,
    options::FindOneOptions,
    results::UpdateResult,
    Client, Collection, Database,
};
use std::{error::Error as StdError, sync::Arc};
use tokio::sync::Mutex;

use crate::{open_meteo::document::FormattedWeatherData, TOKIO_RT};

use super::document::ConversionError;
use lazy_static::lazy_static;
pub struct MongoDb {
    client: Client,
    db: Database,
    collection: Collection<Document>,
}

lazy_static! {
    pub static ref MONGODB_INSTANCE: Arc<Mutex<MongoDb>> = {
        let mongo_db = get_mongo_client();
        Arc::new(Mutex::new(mongo_db))
    };
}

fn get_mongo_client() -> MongoDb {
    TOKIO_RT.block_on(async {
        let mongo_db = MongoDb::new().await.unwrap();
        mongo_db
    })
}

impl MongoDb {
    async fn new() -> Result<Self, Error> {
        let client = Client::with_uri_str("mongodb://localhost:27017").await?;
        let db = client.database("forecast");
        let collection = db.collection("weather_data");
        Ok(Self {
            client,
            db,
            collection,
        })
    }

    pub async fn get_latest_object_doc(
        &self,
        lon: String,
        lat: String,
    ) -> Result<FormattedWeatherData, Box<dyn StdError>> {
        let filter = doc! {
            "lon": lon,
            "lat": lat,
        };

        let options = FindOneOptions::builder()
            .sort(doc! { "datetime": -1 })
            .build();

        let latest_doc = match self.collection.find_one(filter, options).await {
            Ok(Some(doc)) => doc,
            Ok(None) => {
                let err =
                    std::io::Error::new(std::io::ErrorKind::NotFound, "No document was found");
                return Err(Box::new(err));
            }
            Err(err) => return Err(Box::new(err)),
        };

        let doc_to_value: Result<FormattedWeatherData, ConversionError> = latest_doc.try_into();

        Ok(doc_to_value?)
    }

    pub async fn upsert_record(&self, data: Vec<Document>) -> Result<Vec<UpdateResult>, Error> {
        let mut results = Vec::new();

        for document in data {
            if let Some(datetime) = document.get("datetime") {
                let filter = doc! {"datetime": datetime};

                let result = self
                    .collection
                    .update_one(
                        filter,
                        doc! {"$set": document.clone()},
                        mongodb::options::UpdateOptions::builder()
                            .upsert(true)
                            .build(),
                    )
                    .await?;
                results.push(result);
            }
        }
        Ok(results)
    }
}
