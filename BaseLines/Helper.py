from nltk import word_tokenize
from pymongo import MongoClient
import numpy as np
from nltk.util import ngrams

def loadAllText():
    word2index={}
    index2word={}
    allText=[]
    client = MongoClient("mongodb://localhost:27017/")
    db = client["WikiHarvest"]
    col = db["uniques"]
    cursor = col.find({})
    for f in cursor:
        for i in f["result"]:
            try:
                text=i['image_caption']
            except TypeError:
                try:
                    text =i[0]['image_caption']
                except KeyError:
                    text =i[0]['caption']
            allText.append(text)
            sentence = text.lower()
            sequence = word_tokenize(sentence)
            for word in sequence:
                if word not in word2index:
                    word2index[word]=len(word2index)
                    index2word[len(word2index)-1]=word

    return allText,word2index,index2word

def text2WordOverLapVector(text,w2i):
    liste=np.zeros(len(w2i)+1, dtype=int)
    for token in word_tokenize(text.lower()):
        try:
            liste[w2i[token]]=liste[w2i[token]]+1
        except KeyError:
            print("!")
    return liste



#print(len(gg))
#print(len(w2i))
#print(len(i2w))
#print(w2i["carbon"])
#print(i2w[w2i["carbon"]])

