from nltk import word_tokenize
from pymongo import MongoClient
import numpy as np
from nltk.util import ngrams
import math
import collections

def buildmodel(text):
    model = collections.Counter(xgram(text))
    nr_of_ngs = sum(model.values())

    for w in model:
        model[w] = float(model[w]) / float(nr_of_ngs)

    return model

def cosine(a,b):
    return sum([a[k]*b[k] for k in a if k in b]) / (math.sqrt(sum([a[k]**2 for k in a])) * math.sqrt(sum([b[k]**2 for k in b])))

def WordCounter(text):
    text=text.lower()
    text=word_tokenize(text)
    return collections.Counter(text)


def trigram(text):
    try:
        text = '##' + text + '##'
    except TypeError:
        print("!")
    return [text[i:i+3] for i in range(len(text)-2)]

def tgoverlap(t1,t2):
    tg1 = set(trigram(t1))
    tg2 = set(trigram(t2))
    return len(tg1.intersection(tg2)) / len(tg1.union(tg2))

def wordoverlap(t1,t2):
    tg1 = set(WordCounter(t1))
    tg2 = set(WordCounter(t2))
    return len(tg1.intersection(tg2)) / len(tg1.union(tg2))

def ngram(string,n):
    liste = []
    if n < len(string):
        for p in range(len(string) - n + 1) :
            tg = string[p:p+n]
            liste.append(tg)
    return liste

def xgram(string):
    return [w for n in range(1,4) for w in ngram(string.lower(),n)]

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

def text2Vec(text,model):
    vec=np.zeros(300)
    vec_length=1
    for token in word_tokenize(text.lower()):
        if token in model:
            vec=vec+model[token]
            vec_length+=1
    vec=vec/vec_length
    return vec
b=[True, True, False]

print(label_list)


