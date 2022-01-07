import logging

import fasttext
import torch
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.nn import CosineEmbeddingLoss
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from transformers import RobertaModel, RobertaTokenizer

from PyTorch.dataLoader import J_Dataset

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

#model_version = 'roberta-base'
#model = RobertaModel.from_pretrained(model_version, output_attentions=True)
class JBert(nn.Module):
    def __init__(self):
        super(JBert, self).__init__()
        self.model_version = 'roberta-base'
        self.model = RobertaModel.from_pretrained(self.model_version, output_attentions=True)


    def forward(self, x1,x2):
        x1 = self.flatten(x1)
        x2 = self.flatten(x2)

        tf_x1 = self.model(x1)
        tf_x2 = self.model(x2)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        return  cos(tf_x1, tf_x2)


#net = JBert()
criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
roberta=roberta.cuda()
roberta.train()
#roberta.eval()  # disable dropout (or leave in train mode to finetune)
optimizer = optim.Adam(roberta.parameters(), lr=0.001)

#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
print("Load Dataset....")
train_loader=J_Dataset('D:\\PhD\\Notebooks\\captionpairs.xlsx')
print("Dataset loaded!")
counter = []
loss_history = []
iteration_number = 0
NUMBER_EPOCHS = 1
def train_val_dataset(dataset, val_split=0.24):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets
dataset=train_val_dataset(train_loader, val_split=0.80)
dataset2=train_val_dataset(train_loader,val_split=0.20)

train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=64)
eval_dataloader = DataLoader(dataset2["val"], batch_size=64)

train_set=dataset['train']
val_set=dataset2['val']
print("train: ",len(train_set))
print("val: ",len(val_set))
#train_set, val_set = torch.utils.data.random_split(train_loader, [(len(train_loader)/10)*8, (len(train_loader)/10)*2])

#tokenizer = RobertaTokenizer.from_pretrained(model_version)

j=0
writer = SummaryWriter()
loss_fn = torch.nn.L1Loss()

for epoch in range(0, NUMBER_EPOCHS):
    print("Epoch：", epoch, " start.")
    for batch in train_dataloader:
        j=j+1
    #for i, data in enumerate(train_set, 0):

        optimizer.zero_grad()  # clear the calculated grad in previous batch

        batch1 = [roberta.encode(k) for k in batch[0]]
        batch2 = [roberta.encode(k) for k in batch[1]]
        label = [torch.tensor(k).cuda() for k in batch[2]]

        batch1 = [k[:512] for k in batch1]
        batch2=  [k[:512] for k in batch2]

        fet1_all=[roberta.extract_features(k, return_all_hiddens=True) for k in batch1]
        fet2_all=[roberta.extract_features(k, return_all_hiddens=True) for k in batch2]
        for k in fet1_all:
            k[0].requires_grad_(True)
            k[0].retain_grad()
            for q in k:
                q.cuda()

        for k in fet2_all:
            k[0].requires_grad_(True)
            k[0].retain_grad()
            for q in k:
                q.cuda()

        emb_1=[k[0][:,0,:] for k in fet1_all]

        fet1=[k[-1][:,0,:] for k in fet1_all]
        fet2=[k[-1][:,0,:] for k in fet2_all]

        #s1,s2,label=data
        #s1=roberta.encode(s1)         #tokenizer.encode_plus(s1, return_tensors='pt', add_special_tokens=True)
        #s2=roberta.encode(s2)     #tokenizer.encode_plus(s2, return_tensors='pt', add_special_tokens=True)

        #if len(s1)>512:
        #    s1=s1[:512]
        #if len(s2) > 512:
        #    s2 = s2[:512]
        #label=torch.tensor(label)
        #s1=s1.cuda()
        #s2=s2.cuda()
        #label=label.cuda()

        test = "roberta is a transformer"
        test = roberta.encode(test)
        test.cuda()
        feat = roberta.extract_features(test, return_all_hiddens=True)
        writer.add_scalar("firstDim",feat[-1][:, 0, :][0][0],j)

        #features_s1 = roberta.extract_features(s1, return_all_hiddens=True)
        #features_s2 = roberta.extract_features(s2, return_all_hiddens=True)


        #s_token_s1=features_s1[-1][:,0,:]
        #s_token_s2=features_s2[-1][:,0,:]

        #emb_s1=features_s1[0]
        #emb_s2=features_s2[0]

        #emb_s1.retain_grad()
        #emb_s2.retain_grad()

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        #
        cos_val=[]
        loss_list = []

        for i in range(len(fet1)):
            cv=cos(fet1[i],fet2[i])
            cv.retain_grad()
            cos_val.append(cv)
            loss=loss_fn(label[i],cos_val[i])
            loss.backward()
            writer.add_scalar('Loss/train', loss, j)
        #cos_val=cos(s_token_s1,s_token_s2)
        #val=(1+cos_val)/2
        #loss=torch.abs(label-cos_val)
        #loss_list=torch.tensor(loss_list,requires_grad=True)
        #loss = torch.stack(loss_list, dim=0).sum(dim=0).sum(dim=0)
        #loss=torch.sum(loss_list)
        #loss.retain_grad()
        #loss_fn = nn.CosineEmbeddingLoss()
        #loss=loss_fn(s_token_s1[0,:],s_token_s2[0,:],label)

        #loss = CosineEmbeddingLoss(outputs, label)
        #emb_s1.register_hook(lambda grad: print(grad))
        #loss.backward()
        if j%100==0:
            print("Batch #",j,":  ",loss, "     --     ",cos_val)
            #print(cos_val,"   ",loss)


        optimizer.step()

       # if i % 10 == 0:  # show changes of loss value after each 10 batches
        #    # print("Epoch number {}\n Current loss {}\n".format(epoch,loss.item()))
        #    iteration_number += 10
        #    counter.append(iteration_number)
        #    loss_history.append(loss.item())
    # test the network after finish each epoch, to have a brief training result.
    correct_val = 0
    total_val = 0
    truevals=[]
    predictions=[]
    print("EVAL!")

    with torch.no_grad():  # essential for testing!!!!
        for data in val_set:
            s1, s2, label = data

            s1 = roberta.encode(s1)  # tokenizer.encode_plus(s1, return_tensors='pt', add_special_tokens=True)
            s2 = roberta.encode(s2)

            if len(s1) > 512:
                s1 = s1[:512]
            if len(s2) > 512:
                s2 = s2[:512]
            label = torch.tensor(label)
            s1 = s1.cuda()
            s2 = s2.cuda()
            label = label.cuda()
            features_s1 = roberta.extract_features(s1, return_all_hiddens=True)
            features_s2 = roberta.extract_features(s2, return_all_hiddens=True)
            s_token_s1 = features_s1[-1][:, 0, :]
            s_token_s2 = features_s2[-1][:, 0, :]
            cos_val=cos(s_token_s1, s_token_s2)
            #val = (1 + cos_val) / 2
            loss = loss_fn(label,cos_val)
            #print(label,"    ",val)
            predictions.append(cos_val.cpu())
            truevals.append(label.cpu())
            #outputs = net(img0, img1)
            #_, predicted = torch.max(outputs.data, 1)
            #total_val += labels.size(0)
            #correct_val += (predicted == labels).sum().item()
    fpr, tpr, thresholds = metrics.roc_curve(truevals, predictions, pos_label=True)
    auc = metrics.auc(fpr, tpr)
    print("auc: ", auc)
    #print('Accuracy of the network on the', total_val, 'val pairs in', val_famillies,
    #      ': %d %%' % (100 * correct_val / total_val))
    #show_plot(counter, loss_history)






#tokenizer = RobertaTokenizer.from_pretrained(model_version)
#sentence_a = "President Obama delivered the 2011 State of the Union Address on January 25, 2011"

#inputs2 = tokenizer.encode(sentence_a,return_tensors='pt',add_special_tokens=True)
#inputs = tokenizer.encode_plus(sentence_a, return_tensors='pt', add_special_tokens=True)
#input_ids = inputs['input_ids']
#attention = model(input_ids)[-1]
#input_id_list = input_ids[0].tolist() # Batch index 0
#tokens = tokenizer.convert_ids_to_tokens(input_id_list)