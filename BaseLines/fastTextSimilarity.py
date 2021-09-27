import pandas as pd
import numpy as np
from gensim.models.fasttext import load_facebook_vectors
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

import io
from gensim.models import KeyedVectors

from BaseLines.Helper import text2Vec


def load_vectors(fname):
    reloaded_word_vectors = KeyedVectors.load_word2vec_format(fname)
    return reloaded_word_vectors


excel_data = pd.read_excel('D:\\PhD\\Notebooks\\captionpairs.xlsx')
data = pd.DataFrame(excel_data, columns=['s1', 's2', 'same_img','id1','id2'])

#a=cosine_similarity(embeddings[0].reshape(1, -1),embeddings2[0].reshape(1, -1))
cos_sim=[]
cross_sim=[]
y=[]
fastText=load_vectors("D:\\PhD\\fastText_Embeddings\\crawl-300d-2M-subword.vec")
print("Dictionary successfully builded!")

for index,row in data.iterrows():
    if row["same_img"]:
        y.append(1)
    else:
        y.append(0)
s1_list=data['s1'].to_numpy()
s2_list=data['s2'].to_numpy()
emb_s1=[text2Vec(x,fastText) for x in s1_list]
emb_s2=[text2Vec(x,fastText) for x in s2_list]


for i in range(len(emb_s1)):
    cos_sim.append(cosine_similarity(emb_s1[i].reshape(1, -1),emb_s2[i].reshape(1, -1))[0][0])
    #cos_sim.append(cosine(emb_s1[i],emb_s2[i]))
data['cos_sim']=cos_sim
data.to_excel('cos_sim.xlsx')
fpr, tpr, thresholds = metrics.roc_curve(y, cos_sim)
auc=metrics.auc(fpr, tpr)
print("auc: ",auc)
