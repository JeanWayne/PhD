from bson.json_util import dumps
from pymongo import MongoClient

if __name__ == '__main__':
    client = MongoClient("mongodb://localhost:27017/")
    db = client["WikiHarvest"]
    col = db["harvest_1"]
    cursor = col.find({})
    with open('unique.json', 'w') as file:
        file.write('[')
        for document in cursor:
            file.write(dumps(document))
            file.write(',')
        file.write(']')