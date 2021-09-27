from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn import metrics
#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
from MongoDB.Query import getNCaptions, getNCaptionsWithMaxOccurence

model = SentenceTransformer('all-mpnet-base-v2')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
#for sentence, embedding in zip(sentences, embeddings):
#    print("Sentence:", sentence)
#    print("Embedding:", embedding)
#    print("")

#for s in embeddings:
#    for k in embeddings:
#        print(cosine_similarity(s.reshape(1, -1),k.reshape(1, -1))[0])

label,caption=getNCaptionsWithMaxOccurence(n=5000,occurence=5)
embs= model.encode(caption)
auc_scores=[]
print("Number of Captions: ",len(label))
for runs in range(len(label)):
    results=[]
    scores=[]
    auc_label=[]
    selected=runs
    #print(caption[selected])
    for l in range(len(embs)):
        #if l is selected:
        #    continue
        cos=cosine_similarity(embs[l].reshape(1,-1),embs[selected].reshape(1,-1))
        results.append((cos[0][0],label[l]))
        scores.append(cos[0][0])
        if label[l] == label[selected]:
            auc_label.append(1)
        else:
            auc_label.append(0)
        #print(label[l]," : ",cos[0])
    #print("target Label=",label[selected])
    #results.sort(key=lambda tup: tup[0],reverse=True)
    #print(results)
    y =auc_label
    pred = scores
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    auc=metrics.auc(fpr, tpr)
    print("RUN:",runs,"   target label: ",label[runs],"  auc: ",auc)
    auc_scores.append(auc)
print("Numnber of different Captions:",len(label))
print("AUC Mean: ",np.mean(auc_scores))
print("AUC Max: ",np.max(auc_scores))
print("AUC Min: ",np.min(auc_scores))
print("AUC Median: ",np.median(auc_scores))