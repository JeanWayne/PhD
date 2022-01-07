import mmap
import os
from random import shuffle
import seaborn as sns
import numpy
import torch

from gensim.models.keyedvectors import KeyedVectors
from gensim.models import FastText as fText

#fastText_wv = KeyedVectors.load_word2vec_format("D:/PhD/crawl-300d-2M.vec/crawl-300d-2M.vec")
#word_vectors = fastText_wv.wv
#word_vectors.save("fastText.wordvectors")
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import tqdm

print("Loading fastText")
wv = KeyedVectors.load("fastText.wordvectors", mmap='r')
print("Done!")
adj=[]
nouns=[]
def classpred(vec):
    if vec[0]>vec[1]:
        return "Noun"
    else:
        return "Adj"

with open("D:/PhD/english-adjectives.txt") as file:
    for t in file.readlines():
        adj.append(t.strip())
with open("D:/PhD/english-nouns.txt") as file:
    for t in  file.readlines():
        nouns.append(t.strip())
both=adj+nouns
shuffle(both)
y=[]
for i in both:
    if i in adj:
        y.append(torch.tensor([[0.,1.]]))
    elif i in nouns:
        y.append(torch.tensor([[1.,0.]]))
print("Words loaded")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
print("WordsVocab loading")

embeddingDic={}
for word in tqdm(both):
    embeddingDic[word] = torch.tensor([wv[word]], device=device,requires_grad=True)
print("WordsVocab loaded")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(300, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
batch=2
optimizer = optim.Adam(model.parameters(), lr=0.0001)
#print(embeddingDic["map"])
grad_list=[]
w_list=[]
start=len(both)/2
for b in range(0,batch):
    for i in range(len(both)):
        optimizer.zero_grad()
        X = embeddingDic[both[i]]
        #print(X)
        logits = model(X)
        criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
        softm=nn.Softmax()
        y[i]=y[i].cuda()
        #print(logits)
        #print(y[i])
        loss=criterion(logits,y[i])
        loss.backward()
        if i>start:
            grad_list.append(X.grad)
            w_list.append(both[i])
        #pred_probab = #nn.Softmax(dim=1)(logits)
        #y_pred = pred_probab.argmax(1)
        #print(f"Predicted class: {classpred(pred_probab[0])}")
        optimizer.step()
grad_list=[a.cpu().numpy()[0] for a in grad_list]
print(grad_list[:2])
dism=[i for i in range(1,301)]
grad_list=numpy.mean(grad_list,axis=0)
print(grad_list)
#sns.barplot(data=grad_list)
#plt.show()
