import pandas as pd

import numpy as np
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity

from BaseLines.Helper import loadAllText, buildmodel, cosine, trigram, tgoverlap

excel_data = pd.read_excel('D:\\PhD\\Notebooks\\captionpairs.xlsx')
data = pd.DataFrame(excel_data, columns=['s1', 's2', 'same_img','id1','id2'])

#a=cosine_similarity(embeddings[0].reshape(1, -1),embeddings2[0].reshape(1, -1))
cos_sim=[]
cross_sim=[]
y=[]
#gg,w2i,i2w=loadAllText()

print("Dictionary successfully builded!")

for index,row in data.iterrows():
    if row["same_img"]:
        y.append(1)
    else:
        y.append(0)
s1_list=data['s1'].to_numpy()
s2_list=data['s2'].to_numpy()


for i in range(len(s1_list)):
    #cos_sim.append(cosine_similarity(emb_s1[i].reshape(1, -1),emb_s2[i].reshape(1, -1))[0][0])
    cos_sim.append(tgoverlap(s1_list[i],s2_list[i]))
data['cos_sim']=cos_sim
data.to_excel('cos_sim.xlsx')
fpr, tpr, thresholds = metrics.roc_curve(y, cos_sim)
auc=metrics.auc(fpr, tpr)
print("auc: ",auc)
