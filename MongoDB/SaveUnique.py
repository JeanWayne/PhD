import json

from bson.json_util import dumps
from pymongo import MongoClient

if __name__ == '__main__':
    client = MongoClient("mongodb://localhost:27017/")
    db = client["WikiHarvest"]
    col = db["uniques"]
    cursor = col.find({})
    with open('unique.json', 'w') as file:
        file.write('[')
        for document in cursor:
            saveDoc={"id":str(document["_id"]),"results":document["result"]}
            file.write(json.dumps(saveDoc))
            file.write('\n')
        file.write(']')