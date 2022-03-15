import pandas as pd

df = pd.read_excel('D:\\PhD\\Notebooks\\captionpairs.xlsx', sheet_name='Sheet1')
df.drop(['id1','id2'], axis=1, inplace=True)
df.drop(df.columns[0], axis=1, inplace=True)

data = list(zip(df.s1,df.s2,df.same_img))

import random

random.shuffle(data)

split = len(data)//10
train = data[split:]
test = data[:split]

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def bool2int(b):
    if b:
        return 0.95
    else:
        return 0.05

trained_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
train_examples = [InputExample(texts=[a,b], label=bool2int(v)) for a,b,v in train]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(trained_model)

#Tune the model
trained_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

from sentence_transformers import SentenceTransformer
#transf_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
#bert_base_uncase
transf_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

embeddings_a = transf_model.encode([a for a,b,_ in test])
embeddings_b = transf_model.encode([b for a,b,_ in test])

embeddings_a = trained_model.encode([a for a,b,_ in test])
embeddings_b = trained_model.encode([b for a,b,_ in test])

from sklearn.metrics.pairwise import cosine_similarity

def bi_encoder_cos(n):
    e1 = embeddings_a[n]
    e2 = embeddings_b[n]
    sim = cosine_similarity([e1],[e2])
    return sim[0][0]

prediction = [bi_encoder_cos(n) for n in range(len(test))]
truevals = [v for _,_,v in test]

## AUC
import numpy as np
from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(truevals, prediction, pos_label=True)
auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt

plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC using BI_encoder Overlap')
plt.legend(loc="lower right")
plt.show()
##Precision
precision, recall, thresholds = metrics.precision_recall_curve(truevals, prediction, pos_label=True)
plt.plot(recall, precision, color='darkorange', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()
##ACCURACY
def accuracy(pred,labels,threshold):
    correct = 0
    nr = len(pred)
    for i in range(nr):
        if pred[i] > threshold and labels[i]:
            correct+=1
        elif pred[i] <= threshold and not labels[i]:
            correct+=1
    return correct/nr

th = [t/25 for t in range(0,25)]
acc = [accuracy(prediction,truevals,t) for t in th]

plt.plot(th, acc, color='darkorange', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.show()