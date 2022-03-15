import pickle
import gensim
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import io
from joblib import dump, load
from nltk.corpus import wordnet as wn
import nltk
import numpy as np

from gensim.models import KeyedVectors
ft_b="D:/PhD/crawl-300d-2M.vec/crawl-300d-2M.vec"
fname = get_tmpfile("C:/Users/jeanc/PycharmProjects/PhD/Concreteness/fast_model.kv")

with open('C:/Users/jeanc/PycharmProjects/PhD/Concreteness/most_common_ending.pkl', 'rb') as f:
    most_common_ending = pickle.load(f)
with open('C:/Users/jeanc/PycharmProjects/PhD/Concreteness/most_common_porter.pkl', 'rb') as f:
    most_common_porter = pickle.load(f)

reg = load('C:/Users/jeanc/PycharmProjects/PhD/Concreteness/concreteness_SVM.pkl')
model = KeyedVectors.load(fname, mmap='r')

def getPos(w):
    poss=[] # this was two lines above, globel
    posSum=0
    posfreqs = {'n':0,'v':0,'a':0,'p':0,'r':0,'s':0}
    found=False
    for syns in wn.synsets(w):
        p = syns.pos()
        if p not in poss:
            poss.append(p)
        count = posfreqs.get(p,0)
        for l in syns.lemmas():
            count += l.count()
            if count>0:
                found=True
        posfreqs[p] = count
    if found:
        for fr in posfreqs:
            posSum+=posfreqs[fr]
        pos=[posfreqs['n']/posSum,posfreqs['v']/posSum,posfreqs['a']/posSum,posfreqs['p']/posSum,posfreqs['r']/posSum,posfreqs['s']/posSum]
    else:
        tags = nltk.pos_tag([w])
        tag = tags[0][1]
        if tag[0] == 'R':
            pos = 'r'
            pos=[0,0,0,0,1,0]
        elif tag[0] == 'J':
            pos = 'a'
            pos=[0,0,1,0,0,0]
        elif tag[0] == 'V':
            pos = 'v'
            pos=[0,1,0,0,0,0]
        elif tag == 'IN':
            pos = 'p'
            pos=[0,0,0,1,0,0]
        else:
            pos = 'n'
            pos=[1,0,0,0,0,0]
    return pos

def getAllPostfix(s):
    if len(s)>3:
        return [s[-1:],s[-2:],s[-3:],s[-4:]]
    elif len(s)==3:
        return [s[-1:],s[-2:],s[-3:]]
    elif len(s)==2:
        return [s[-1:],s[-2:]]
    elif len(s)==1:
        return [s]
    else:
        return [""]

def getPostfixVec(s,mc,dim=100):
    vec=[0]*dim
    Post=getAllPostfix(s)
    #Post=Func(s)
    for i in range(dim):
        if mc[i] in Post:
            vec[i]=1
    return vec
def getVecFromPos(s):
    vec=[]
    if s=='a':
        vec=[1,0,0,0,0,0]
    if s=='n':
        vec=[0,1,0,0,0,0]
    if s=='p':
        vec=[0,0,1,0,0,0]
    if s=='r':
        vec=[0,0,0,1,0,0]
    if s=='s':
        vec=[0,0,0,0,1,0]
    if s=='v':
        vec=[0,0,0,0,0,1]
    return vec#np.asarray(vec)

def getFeature(w,emb=True,POS=True,Post=True):
    vect=[]
    vect=np.asarray(vect)

    if emb:
        vect=np.append(vect,model[w])
    if POS:
        vect=np.append(vect,getPos(w))
    if Post:
        vect=np.append(vect,getPostfixVec(w,most_common_ending,dim=200))
    return vect

#model = KeyedVectors.load_word2vec_format(ft_b)
#print(model["house"])

#model.save(fname)

def getConcreteness(word):
    try:
        fvec = getFeature(word, emb=True, POS=True, Post=True)
        ret = reg.predict([fvec])
    except KeyError:
        return 3
    return ret


