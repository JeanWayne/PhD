import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import json

with open('idf.json') as json_file:
    idf = json.load(json_file)
conc={}
with open('concreteness.csv','r') as csv_file:
    for line in csv_file.readlines():
        datum=line.split('\t')
        conc[datum[0]]=float(datum[1].strip())


stop_words = set(stopwords.words('english'))
porter = PorterStemmer()
lemma = nltk.wordnet.WordNetLemmatizer()

#print(stop_words)
added_stopwords={",",".","#","%","&"}



def Overlap(cap1,cap2):
    #tonkenize
    cap1=word_tokenize(cap1)
    cap2=word_tokenize(cap2)
    #remove stopwords
    cap1_nsw=[]
    cap2_nsw=[]
    for w in cap1:
        if w not in stop_words and w not in added_stopwords:
            cap1_nsw.append(w.lower())
    for w in cap2:
        if w not in stop_words and w not in added_stopwords:
            cap2_nsw.append(w.lower())
    cap1=cap1_nsw
    cap2=cap2_nsw

    #print(cap1)
    #print(cap2)
    #print(Jaccard_Similarity(cap1,cap2))
    #print(Jaccard_IDF(cap1,cap2))
    #print(Jaccard_Conc(cap1,cap2))

def Jaccard_overlap(cap1,cap2):
    #tonkenize
    cap1=word_tokenize(cap1)
    cap2=word_tokenize(cap2)
    #remove stopwords
    cap1_nsw=[]
    cap2_nsw=[]
    for w in cap1:
        if w not in stop_words and w not in added_stopwords:
            #cap1_nsw.append(porter.stem(w.lower()))
            cap1_nsw.append(lemma.lemmatize(w.lower()))
    for w in cap2:
        if w not in stop_words and w not in added_stopwords:
            #cap2_nsw.append(porter.stem(w.lower()))
            cap2_nsw.append(lemma.lemmatize(w.lower()))
    cap1=cap1_nsw
    cap2=cap2_nsw

    #return Jaccard_Similarity(cap1,cap2)
    return Jaccard_IDF(cap1,cap2)


def Jaccard_Similarity(doc1, doc2):
    words_doc1 = set(doc1)
    words_doc2 = set(doc2)
    intersection = words_doc1.intersection(words_doc2)
    union = words_doc1.union(words_doc2)
    return float(len(intersection)) / len(union)

def Jaccard_weight(doc1,doc2):
    words_doc1 = set(doc1)
    words_doc2 = set(doc2)
    countDoc1= nltk.Counter(doc1)
    countDoc2= nltk.Counter(doc2)
    union = words_doc1.union(words_doc2)
    top_sum=0
    low_sum=0.0
    if doc1==doc2:
        return 0
    if len(doc1)<3 or len(doc2)<3:
        return 0
    for w in union:
        if w in idf:
            top_sum+=min(countDoc1[w]*idf[w],countDoc2[w]*idf[w])
            low_sum+=max(countDoc1[w]*idf[w],countDoc2[w]*idf[w])
        else:
            print(w)
    if low_sum==0:
        return 0
    return top_sum/low_sum

def intersect(doc1,doc2):
    words_doc1 = set(doc1)
    words_doc2 = set(doc2)
    intersection = words_doc1.intersection(words_doc2)
    union = words_doc1.union(words_doc2)
    return intersection, union

def Jaccard_IDF(doc1,doc2):
    intersection, union= intersect(doc1,doc2)
    lenIntersection = len(intersection)
    lenUnion = len(union)
    for w in intersection:
        if w in idf:
            lenIntersection+=idf[w]
    for w in union:
        if w in idf:
            lenUnion+=idf[w]
    return float(lenIntersection/lenUnion)

def Jaccard_Conc(doc1,doc2):
    intersection, union= intersect(doc1,doc2)
    lenIntersection = len(intersection)
    lenUnion = len(union)
    for w in intersection:
        if w in conc:
            lenIntersection+=conc[w]
    for w in union:
        if w in conc:
            lenUnion+=conc[w]

    return float(lenIntersection/lenUnion)

#a="You are blessed. Make the most of it."
#b="Former Governor and Political commentator Mike Huckabee of Arkansas"
#Overlap(a,b)