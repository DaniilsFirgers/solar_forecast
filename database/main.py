from pymongo import MongoClient
from config.database import DbConfig, MONGO_DB_HOST, MONGO_DB_PORT, MONGO_USERNAME, MONGO_PASSWORD
from data.transform import Datapoint


class MongoDBHandler:
    db_config: DbConfig
    client: MongoClient

    def __init__(self, db_config: DbConfig):
        self.client = None
        self.db_config = db_config

    def connect(self):
        # Connect to MongoDB
        try:
            if self.client is not None:
                return
            self.client = MongoClient(
                self.db_config.host, self.db_config.port, username=self.db_config.username, password=self.db_config.password)
            print("Connected to MongoDB successfully!")
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")

    def disconnect(self):
        # Disconnect from MongoDB
        if self.client:
            self.client.close()
            print("Disconnected from MongoDB.")

    def retrieve_data(self, database_name, collection_name, filter_query={}):
        # Retrieve data from MongoDB with optional filter
        try:
            if self.client is None:
                self.connect()
            db = self.client[database_name]
            collection = db[collection_name]
            result: list[Datapoint] = collection.find(filter_query)
            formatted_result = [Datapoint(**datapoint).to_object()
                                for datapoint in result]
            return list(formatted_result)
        except Exception as e:
            print(f"Error retrieving data from MongoDB: {e}")
        finally:
            self.disconnect()


mongo_config = DbConfig(
    host=MONGO_DB_HOST,
    port=MONGO_DB_PORT,
    username=MONGO_USERNAME,
    password=MONGO_PASSWORD
)

mongo_handler = MongoDBHandler(mongo_config)
