use mongodb::{
    bson::{doc, Document},
    error::Error,
    results::UpdateResult,
    Client, Collection, Database,
};
pub struct MongoDb {
    client: Client,
    db: Database,
    collection: Collection<Document>,
}

impl MongoDb {
    pub async fn new() -> Result<Self, Error> {
        let client = Client::with_uri_str("mongodb://localhost:27017").await?;
        let db = client.database("forecast");
        let collection = db.collection("weather_data");
        Ok(Self {
            client,
            db,
            collection,
        })
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
