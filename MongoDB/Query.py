import pymongo


client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["WikiHarvest"]
col = db["uniques"]


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
                try:
                    caption.append(t[0]["image_caption"])
                except:
                    caption.append(t[0]['caption'])
    return label,caption

def getNCaptionsWithMaxOccurence(n=1,occurence=10):
    label=[]
    caption=[]
    l=0
    for f in col.find().limit(n):
        if f['result_size']<occurence:
            for g in f['result']:
                l=l+1
                label.append(l)
                caption.append(g["image_caption"])

    return label,caption
