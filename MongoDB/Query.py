import pymongo


client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["WikiHarvest"]
col = db["harvest_1"]


def getNCaptions(n):
    label=[]
    caption=[]
    l=0
    for f in col.find().limit(n):
        #print("Docu--------------")
        for g in f['images']:
            l=l+1
            for t in g['result']:
                label.append(l)
                caption.append(t[0]["image_caption"])
    return label,caption
