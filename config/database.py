import dataclasses

MONGO_DB_HOST = 'localhost'
MONGO_DB_PORT = 27017
MONGO_USERNAME = None
MONGO_PASSWORD = None


DB_NAME = 'customer_management'
COLLECTION_NAME = 'power_meter_history'


@dataclasses.dataclass
class DbConfig:
    host: str
    port: int
    username: str
    password: str
