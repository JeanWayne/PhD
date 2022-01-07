import os
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
from torch.utils.data import Dataset
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def bool2int(b):
    if b:
        return 1.
    else:
        return -1.

class J_Dataset(Dataset):
    def __init__(self, excel_file, target_transform=None):
        excel_data = pd.read_excel(excel_file)
        df = pd.DataFrame(excel_data, columns=['s1', 's2', 'same_img', 'id1', 'id2'])
        data = list(zip(df.s1, df.s2, df.same_img))
        self.data = [[a,b,bool2int(c)] for a,b,c in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class J_Dataset_idx(Dataset):
    def __init__(self, excel_file, target_transform=None):
        excel_data = pd.read_excel(excel_file)
        df = pd.DataFrame(excel_data, columns=['s1', 's2', 'same_img', 'id1', 'id2'])
        data = list(zip(df.s1, df.s2, df.same_img))
        self.data = [{"a":tokenizer(a,return_tensors="pt").to('cuda'),"b":tokenizer(b,return_tensors="pt").to('cuda'),"label" :bool2int(c)} for a,b,c in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
