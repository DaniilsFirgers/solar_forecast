use mongodb::{
    bson::{doc, Document},
    error::Error,
    options::FindOneOptions,
    results::UpdateResult,
    Client, Collection, Database, IndexModel,
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

        let datetime_index_model = IndexModel::builder().keys(doc! { "datetime": 1 }).build();

        let object_name_index_model = IndexModel::builder()
            .keys(doc! { "object_name": 1 })
            .build();

        collection.create_index(datetime_index_model, None).await?;
        collection
            .create_index(object_name_index_model, None)
            .await?;
        Ok(Self {
            client,
            db,
            collection,
        })
    }

    pub async fn get_latest_object_doc(
        &self,
        object_name: &String,
    ) -> Result<FormattedWeatherData, Box<dyn StdError>> {
        let filter = doc! {
            "object_name": object_name
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
            Err(err) => {
                return Err(Box::new(err));
            }
        };

        let doc_to_value: Result<FormattedWeatherData, ConversionError> = latest_doc.try_into();

        Ok(doc_to_value?)
    }

    pub async fn upsert_record(&self, data: Vec<Document>) -> Result<Vec<UpdateResult>, Error> {
        let mut results = Vec::new();

        for document in data {
            if let (Some(datetime), Some(object_name)) =
                ((document.get("datetime")), (document.get("object_name")))
            {
                let filter = doc! {"datetime": datetime, "object_name": object_name};

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
