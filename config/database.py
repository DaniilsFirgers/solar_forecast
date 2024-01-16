import dataclasses

MONGO_DB_HOST = 'localhost'
MONGO_DB_PORT = 27017
MONGO_USERNAME = None
MONGO_PASSWORD = None


PRODUCTION_HISTORY_DB_NAME = 'customer_management'
PRODUCTION_HISTORY_COLLECTION_NAME = 'power_meter_history'


@dataclasses.dataclass
class DbConfig:
    host: str
    port: int
    username: str
    password: str
