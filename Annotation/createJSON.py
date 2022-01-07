import pymongo
from random import randint
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import json
import random

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

def loadDocumentsFromMongo(number,desired):
    arr = []
    for doc in col.aggregate([{'$sample': {'size': number}}]):
        writeDoc = {}
        writeDoc['ID'] = str(doc["_id"])
        randomOccurenceIndex = randint(0, doc['result_size'] - 1)
        writeDoc['URL'] = doc['wc_URL']
        try:
            if 'image_caption' not in doc['result'][randomOccurenceIndex]:
                writeDoc['caption'] = doc['result'][randomOccurenceIndex][0]['image_caption']
                writeDoc['wikipediaURL'] = doc['result'][randomOccurenceIndex][0]['image_url']
            elif "image_caption" in doc['result'][randomOccurenceIndex]:
                writeDoc['caption'] = doc['result'][randomOccurenceIndex]['image_caption']
                writeDoc['wikipediaURL'] = doc['result'][randomOccurenceIndex]['image_url']
            else:
                print("Error!")
                continue
                # writeDoc['caption'] = doc['result'][randomOccurenceIndex][0]['caption']
                # writeDoc['wikipediaURL'] = doc['result'][randomOccurenceIndex][0]['found']
        except KeyError:
            print("!")
            continue
        split = word_tokenize(writeDoc['caption'])
        split = [s for s in split if s not in stoplist]
        split = [s for s in split if len(s) > 1]
        writeDoc['tokens'] = split
        writeDoc['selected'] = []
        if len(split) < 5 or "List of" in writeDoc['caption'] or len(writeDoc["caption"]) > 511:
            continue
        arr.append(writeDoc)
        if len(arr)>=desired:
            break
    return arr,len(arr)

sample_per_user=300
number_of_users=10
shared_items=30
number_of_items_toQuery=(sample_per_user * number_of_users) + ((shared_items/2) * number_of_users)
stoplist = set(stopwords.words('english') + list(punctuation)+ ["``","''","'s"])
listOfDocs=[]
sharelist=[]

count=0
allDocuments,_=loadDocumentsFromMongo(8000,number_of_items_toQuery)
for doc in allDocuments:
    if count<(shared_items/2)*number_of_users:
        sharelist.append(doc)
    else:
        listOfDocs.append(doc)
    count += 1
temp=[]
IDS=[it['ID'] for it in sharelist]
print(IDS)
for item in sharelist:
    temp.append(item)
    temp.append(item)
sharelist=temp
random.shuffle(sharelist)



for k in range(number_of_users):
    writeArr=[]
    with open("Files/File_"+str(k)+".json", mode="w",encoding="utf-8") as file:
        for i in range(sample_per_user):
            writeArr.append(listOfDocs.pop(0))
        for i in range(shared_items):
            if sharelist[0] not in writeArr:
                writeArr.append(sharelist.pop(0))
            else:
                writeArr.append(sharelist.pop(1))

        json.dump(writeArr, file,indent=4)


print("!")


# for i in range(number_of_users):
#     with open("Label_File_#"+str(1)+".json", mode="w",encoding="utf-8") as file:
#         arr=[]
#         for doc in col.aggregate([{'$sample': {'size': sample_per_user*number_of_users}}]):
#             writeDoc={}
#             writeDoc['ID']=str(doc["_id"])
#             randomOccurenceIndex=randint(0,doc['result_size']-1)
#             writeDoc['URL']=doc['wc_URL']
#             writeDoc['caption']=doc['result'][randomOccurenceIndex]['image_caption']
#             writeDoc['wikipediaURL']=doc['result'][randomOccurenceIndex]['image_url']
#             split=word_tokenize(writeDoc['caption'])
#             split=[s for s in split if s not in stoplist]
#             split=[s for s in split if len(s)>1]
#             writeDoc['tokens']=split
#             writeDoc['selected']=[]
#             if len(split)<5 or "List of" in writeDoc['caption'] or len(writeDoc)["caption"]>511:
#                 continue
#             if len(arr)>sample_per_user*number_of_users:
#                 json.dump(arr, file, indent=4)
#                 break
#             arr.append(writeDoc)
#         json.dump(arr, file,indent=4)


