import dataclasses

MONGO_DB_HOST = 'localhost'
MONGO_DB_PORT = 27017
MONGO_USERNAME = None
MONGO_PASSWORD = None


PRODUCTION_HISTORY_DB_NAME = 'forecast'
PRODUCTION_HISTORY_COLLECTION_NAME = 'production_history'
PRODUCTION_WEATHER_COLLECTION_NAME = 'weather_data'


@dataclasses.dataclass
class DbConfig:
    host: str
    port: int
    username: str
    password: str
