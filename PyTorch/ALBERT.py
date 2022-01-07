import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import os
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

from PyTorch.dataLoader import J_Dataset


class MNLIDataBert(Dataset):

  def __init__(self, train_df, val_df):
    #self.label_dict = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    self.label_dict= {True:1,False:0}
    self.train_df = train_df
    self.val_df = val_df

    self.base_path = '/content/'
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) # Using a pre-trained BERT tokenizer to encode sentences
    #self.train_data = None
    #self.val_data = None
    self.init_data()

  def init_data(self):
      try:
        self.train_data = self.load_data(self.train_df)
        self.val_data = self.load_data(self.val_df)
      except:
          print("")

  def load_data(self, df):
    MAX_LEN = 512
    token_ids = []
    mask_ids = []
    seg_ids = []
    y = []

    data = list(zip(df.s1, df.s2, df.same_img))
    #train_loader = J_Dataset('D:\\PhD\\Notebooks\\captionpairs.xlsx')
    premise_list=[x[0][:512] for x in data]
    hypothesis_list=[x[1][:512] for x in data]
    label_list=[x[2] for x in data]
    #premise_list = [df['s1'].to_list()]
    #hypothesis_list = df['s2'].to_list()
    #label_list = df['same_img'].to_list()

    for (premise, hypothesis, label) in zip(premise_list, hypothesis_list, label_list):
      premise_id = self.tokenizer.encode(premise, add_special_tokens = False)
      hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False)
      pair_token_ids = [self.tokenizer.cls_token_id] + premise_id + [self.tokenizer.sep_token_id] + hypothesis_id + [self.tokenizer.sep_token_id]
      premise_len = len(premise_id)
      hypothesis_len = len(hypothesis_id)

      segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
      attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

      token_ids.append(torch.tensor(pair_token_ids))
      seg_ids.append(segment_ids)
      mask_ids.append(attention_mask_ids)
      y.append(self.label_dict[label])

    token_ids = pad_sequence(token_ids, batch_first=True)
    mask_ids = pad_sequence(mask_ids, batch_first=True)
    seg_ids = pad_sequence(seg_ids, batch_first=True)
    y = torch.tensor(y)
    dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)
    print(len(dataset))
    return dataset

  def get_data_loaders(self, batch_size=32, shuffle=True):
    train_loader = DataLoader(
      self.train_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    val_loader = DataLoader(
      self.val_data,
      shuffle=shuffle,
      batch_size=batch_size
    )

    return train_loader, val_loader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def multi_acc(y_pred, y_test):
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
    return acc


def train(model, train_loader, val_loader, optimizer):
    total_step = len(train_loader)

    for epoch in range(EPOCHS):
        start = time.time()
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(train_loader):
            optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)

            loss, prediction = model(pair_token_ids,
                                     token_type_ids=seg_ids,
                                     attention_mask=mask_ids,
                                     labels=labels).values()

            acc = multi_acc(prediction, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_acc += acc.item()

        train_acc = total_train_acc / len(train_loader)
        train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_val_acc = 0
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in enumerate(val_loader):
                optimizer.zero_grad()
                pair_token_ids = pair_token_ids.to(device)
                mask_ids = mask_ids.to(device)
                seg_ids = seg_ids.to(device)
                labels = y.to(device)

                loss, prediction = model(pair_token_ids,
                                         token_type_ids=seg_ids,
                                         attention_mask=mask_ids,
                                         labels=labels).values()

                acc = multi_acc(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc += acc.item()

        val_acc = total_val_acc / len(val_loader)
        val_loss = total_val_loss / len(val_loader)
        end = time.time()
        hours, rem = divmod(end - start, 3600)
        minutes, seconds = divmod(rem, 60)

        print(
            f'Epoch {epoch + 1}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}')
        print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

excel_file='D:\\PhD\\Notebooks\\captionpairs.xlsx'
excel_data = pd.read_excel(excel_file)
df_ = pd.DataFrame(excel_data, columns=['s1', 's2', 'same_img', 'id1', 'id2'])
df_train = df_.iloc[:, :500]
df_test  = df_.iloc[:, 500:]
mnli_dataset = MNLIDataBert(df_train, df_test)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)
print(f'The model has {count_parameters(model):,} trainable parameters')

train_loader, val_loader = mnli_dataset.get_data_loaders(batch_size=16)

train(model, train_loader, val_loader, optimizer)






import time

EPOCHS = 5

