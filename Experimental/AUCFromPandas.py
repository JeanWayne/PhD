import pandas as pd

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn import metrics
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers.cross_encoder import CrossEncoder

from MongoDB.Query import getNCaptions, getNCaptionsWithMaxOccurence

model = SentenceTransformer('all-mpnet-base-v2')
model_Cross = CrossEncoder('distilbert-base-uncased')
#Our sentences we like to encode


#Sentences are encoded by calling model.encode()
#embeddings = model.encode(sentences)
#embeddings2 = model.encode(sentences2)

excel_data = pd.read_excel('D:\\PhD\\Notebooks\\captionpairs.xlsx')
data = pd.DataFrame(excel_data, columns=['s1', 's2', 'same_img','id1','id2'])

#a=cosine_similarity(embeddings[0].reshape(1, -1),embeddings2[0].reshape(1, -1))
cos_sim=[]
cross_sim=[]
y=[]
for index,row in data.iterrows():
    if row["same_img"]:
        y.append(1)
    else:
        y.append(0)
s1_list=data['s1'].to_numpy()
s2_list=data['s2'].to_numpy()
emb_s1=model.encode(s1_list)
emb_s2=model.encode(s2_list)

for i in range(len(emb_s1)):
    cos_sim.append(cosine_similarity(emb_s1[i].reshape(1, -1),emb_s2[i].reshape(1, -1))[0][0])
    cross_sim.append(model_Cross.predict([s1_list[i],s2_list[i]]))
data['cos_sim']=cos_sim
data.to_excel('cos_sim.xlsx')
fpr, tpr, thresholds = metrics.roc_curve(y, cos_sim)
auc=metrics.auc(fpr, tpr)
print("auc: ",auc)
fpr, tpr, thresholds = metrics.roc_curve(y, cross_sim)
auc=metrics.auc(fpr, tpr)
print("cross auc: ",auc)