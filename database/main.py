from pymongo import MongoClient
from config.database import DbConfig, MONGO_DB_HOST, MONGO_DB_PORT, MONGO_USERNAME, MONGO_PASSWORD
from data_handling.transform import Datapoint
import pandas as pd


class MongoDBHandler:
    db_config: DbConfig
    client: MongoClient

    def __init__(self, db_config: DbConfig):
        self.client = None
        self.db_config = db_config
        self.client = MongoClient(
            self.db_config.host, self.db_config.port, username=self.db_config.username, password=self.db_config.password)
        print("Connected to MongoDB.")

    def disconnect(self):
        if self.client:
            self.client.close()
            print("Disconnected from MongoDB.")

    def retrieve_data(self, database_name, collection_name, filter_query={}):
        # Retrieve data from MongoDB with optional filter
        try:
            db = self.client[database_name]
            collection = db[collection_name]
            result: list[Datapoint] = collection.find(filter_query)
            formatted_result = [Datapoint(**datapoint).to_object()
                                for datapoint in result]
            df = pd.DataFrame(formatted_result).sort_values("start_time")
            df.set_index("start_time", inplace=True)
            return df
        except Exception as e:
            print(f"Error retrieving data from MongoDB: {e}")


mongo_config = DbConfig(
    host=MONGO_DB_HOST,
    port=MONGO_DB_PORT,
    username=MONGO_USERNAME,
    password=MONGO_PASSWORD
)

mongo_handler = MongoDBHandler(mongo_config)
