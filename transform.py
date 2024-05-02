from pymongo import MongoClient

# MongoDB connection
uri = "mongodb://localhost:27017/"
client = MongoClient(uri)
db_name = 'forecast'
collection_name = 'production_data_tmp'
db = client[db_name]
collection = db[collection_name]

# Mapping dictionary
mapping_dict = {
    "SADALES_TIKLS_lv_producer_43X-STJ02620895C_43Z-STO01766085R_12624502": "A",
    "SADALES_TIKLS_lv_producer_43X-STJ02101362F_43Z-STO01848266I_12587328": "C",
    "SADALES_TIKLS_lv_producer_43X-STJ02625769T_43Z-STO01766477A_12628530": "B"
}

# Process documents
for document in collection.find():
    # Extract and transform the necessary fields
    id = document.get('_id')
    start_time = document.get('start_time')
    value = document.get('value')
    reference = document.get('reference')
    object_name = mapping_dict.get(reference)

    # Skip if mapping is not found (or handle as needed)
    if object_name is None:
        continue

    # Create new formatted document
    new_document = {
        'start_time': start_time,
        'value': value,
        'object_name': object_name
    }

    update_operation = {
        "$unset": {
            "reference": "",
            "interval": "",
            "area": "",
            "object_type": "",
            "internal_type": "",
            "production": "",
        }
    }

    # Upsert the new document
    collection.update_one(
        # Query for existing document
        {'_id': id},
        update=update_operation,  # New document content
        # upsert=True  # Insert if does not exist
    )
