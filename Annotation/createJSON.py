from collections import Counter

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

sample_per_user=270
number_of_users=10
shared_items=30
unique_shared_items_count=shared_items
number_of_items_toQuery=(sample_per_user * number_of_users) + unique_shared_items_count

stoplist = set(stopwords.words('english') + list(punctuation)+ ["``","''","'s"])
listOfDocs=[]
sharelist=[]

count=0
allDocuments,_=loadDocumentsFromMongo(8000,number_of_items_toQuery)
#randomDocuments,_=loadDocumentsFromMongo(shared_items*unique_shared_items_count,unique_shared_items_count)

#for doc in randomDocuments:
#    sharelist.append(doc)
count=0
for doc in allDocuments:
    if len(sharelist)<shared_items:
        sharelist.append(doc)
    else:
        listOfDocs.append(doc)
    count+=1

random.shuffle(sharelist)

added=[]
toWrite=[]
IDS=[]
for k in range(number_of_users):
    writeArr=[]
    for i in range(sample_per_user):
        writeArr.append(listOfDocs.pop(0))
    toWrite.append(writeArr)

IDS=[w["ID"] for w in sharelist]

for j in range(number_of_users):
    toWrite[j].extend(sharelist)


for k in range(len(toWrite)):
    with open("Files/DataSet"+str(k)+".json", mode="w",encoding="utf-8") as file:
        random.shuffle(toWrite[k])
        if len(toWrite[k])!=sample_per_user+shared_items:
            print("Error in Arr size")
        else:
            json.dump(toWrite[k], file,indent=4)

with open("unique_IDS.txt","w") as f:
    for d in IDS:
        f.write(d+"\n")
print(IDS)

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


