# Importing the libraries needed
import os
from datetime import datetime

from torch.optim.lr_scheduler import StepLR

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json


from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import RobertaModel, RobertaTokenizer, BertTokenizer, BertModel, DistilBertTokenizer, DistilBertModel
import logging
logging.basicConfig(level=logging.ERROR)


# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 12
VALID_BATCH_SIZE = 8
# EPOCHS = 1
LEARNING_RATE = 2e-05
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True, do_lower_case=True)
#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base', truncation=True, do_lower_case=True)
def bool2int(b):
    if b:
        return 1.
    else:
        return -1.

class SentimentData(Dataset):
    def __init__(self, tokenizer, max_len,excel_file):
        excel_data = pd.read_excel(excel_file)
        df = pd.DataFrame(excel_data, columns=['s1', 's2', 'same_img', 'id1', 'id2'])
        data = list(zip(df.s1, df.s2, df.same_img))
        self.data = [[a, b, bool2int(c)] for a, b, c in data]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s1=self.data[index][0]
        s2=self.data[index][1]
        label=self.data[index][2]
        inputs = self.tokenizer.encode_plus(
            s1,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        inputs2 = self.tokenizer.encode_plus(
            s2,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        ids2 = inputs2['input_ids']
        mask2 = inputs2['attention_mask']
        token_type_ids2 = inputs2["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long,requires_grad=False),
            'mask': torch.tensor(mask, dtype=torch.long,requires_grad=False),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long,requires_grad=False),
            'ids2': torch.tensor(ids2, dtype=torch.long,requires_grad=False),
            'mask2': torch.tensor(mask2, dtype=torch.long,requires_grad=False),
            'token_type_ids2': torch.tensor(token_type_ids2, dtype=torch.long,requires_grad=False),
            'targets': torch.tensor(label, dtype=torch.float,requires_grad=False)
        }

dataset=SentimentData(tokenizer,MAX_LEN,'D:\\PhD\\Notebooks\\captionpairs.xlsx')

def train_val_dataset(dataset, val_split=0.05):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets
dataset=train_val_dataset(dataset, val_split=0.15)

train_dataloader = DataLoader(dataset["train"], shuffle=True,batch_size=TRAIN_BATCH_SIZE)
eval_dataloader = DataLoader(dataset["val"],batch_size=TRAIN_BATCH_SIZE)

train_size = 0.85
#traindata=DataLoader(dataset)

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        #self.l1 = BertModel.from_pretrained("bert-base-uncased")
        self.l1 = RobertaModel.from_pretrained("roberta-base")
        #self.l1 = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.sm = torch.nn.Softmax()
        self.pre_classifier = torch.nn.Linear(2*768, 128)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(128, 2)
        self.transform= torch.nn.Identity()

        self.cosine = nn.CosineSimilarity()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    def forward(self, input_ids, attention_mask, token_type_ids,input_ids2, attention_mask2, token_type_ids2):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)#, token_type_ids=token_type_ids)
        output_2 = self.l1(input_ids=input_ids2, attention_mask=attention_mask2)#, token_type_ids=token_type_ids2)
        hidden_state_1 = output_1[0]
        hidden_state_2 = output_2[0]
        pooler_1 = hidden_state_1[:, 0,:]
        pooler_2= hidden_state_2[:, 0,:]
        cat=self.cosine(pooler_1,pooler_2)
        cat=self.transform(cat)


        #cat=self.relu(cat)
        ##CrossEncoderApproach
        #cat=torch.cat((pooler_1,pooler_2),1)
        #cat=self.pre_classifier(cat)
        #cat=self.dropout(cat)
        #cat=self.classifier(cat)
        #cat=self.sm(cat)
        #output=self.cos(hidden_state_2,hidden_state_1)
        #output=self.cos(pooler_1,pooler_2)
        return cat

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: ",device)
model = RobertaClass()
print(model)
model.train()
model.to(device)

# Creating the loss function and optimizer
#loss_function = torch.nn.CrossEntropyLoss()
#loss_function = torch.nn.L1Loss()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(params =  model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=7000, gamma=0.5)
print(model)
print(model.parameters())
def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

writer = SummaryWriter()

def train(epoch):
    print("LenTrain: ",len(train_dataloader))
    print("LenVal: ",len(eval_dataloader))
    tr_loss = 0
    tr_loss_last_track=0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    track_size = 500
    j=0
    eval_j=0
    for _, data in tqdm(enumerate(train_dataloader, 0)):
        j=j+1
        model.train()
        #model.zero_grad()
        optimizer.zero_grad()
        #optimizer.train()
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

        ids2 = data['ids2'].to(device, dtype=torch.long)
        mask2 = data['mask2'].to(device, dtype=torch.long)
        token_type_ids2 = data['token_type_ids2'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids,ids2, mask2, token_type_ids2)

        loss = loss_function(outputs,targets)#(outputs, targets)
        writer.add_scalar("loss", loss, j)
        #gra=torch.autograd.grad(model.l1[0],ids)
        tr_loss += loss.item()
        tr_loss_last_track+=loss.item()
        #big_val, big_idx = torch.max(outputs.data, dim=1)
        #n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % track_size == 0:
            loss_step = tr_loss / nb_tr_steps
            loss_track=tr_loss_last_track/track_size
            #accu_step = (n_correct * 100) / nb_tr_examples
            print(f"Training Loss per {track_size} steps: {loss_step}")
            print(f"Training Loss per track: {loss_track}")
            tr_loss_last_track=0
            #print(f"Training Accuracy per 500 steps: {accu_step}")

        loss.backward()
        # # When using GPU
        optimizer.step()
        scheduler.step()

    truevals = []
    predictions = []
    print("EVAL!")
    with torch.no_grad():  # essential for testing!!!!
        for _, data in tqdm(enumerate(eval_dataloader, 0)):
            eval_j = eval_j + 1
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)

            ids2 = data['ids2'].to(device, dtype=torch.long)
            mask2 = data['mask2'].to(device, dtype=torch.long)
            token_type_ids2 = data['token_type_ids2'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids, ids2, mask2, token_type_ids2)
            loss = loss_function(outputs, targets)  # (outputs, targets)
            writer.add_scalar("EVAL loss", loss, eval_j)

            predictions.append(outputs.cpu().numpy()[0])
            truevals.append(targets.cpu().numpy()[0])


    fpr, tpr, thresholds = metrics.roc_curve(truevals, predictions, pos_label=True)
    auc = metrics.auc(fpr, tpr)
    print("AUC: ", auc)
    save_path="./Model/"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+"_Epoch_"+str(epoch)+"AUC_"+str(auc)
    model.l1.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    #print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    #epoch_loss = tr_loss / nb_tr_steps
    #epoch_accu = (n_correct * 100) / nb_tr_examples
    #print(f"Training Loss Epoch: {epoch_loss}")
    #print(f"Training Accuracy Epoch: {epoch_accu}")
    return

EPOCHS = 9
for epoch in range(EPOCHS):
    train(epoch)