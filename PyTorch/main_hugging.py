from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from transformers import RobertaTokenizer, RobertaModel

from PyTorch.dataLoader import J_Dataset, J_Dataset_idx

#tokenized_datasets = tokenized_datasets.remove_columns(["text"])
#tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
#tokenized_datasets.set_format("torch")

#small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
#small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

train_loader=J_Dataset_idx('D:\\PhD\\Notebooks\\captionpairs.xlsx')
print("Dataset loaded!")
def train_val_dataset(dataset, val_split=0.24):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets
dataset=train_val_dataset(train_loader, val_split=0.90)
dataset2=train_val_dataset(train_loader,val_split=0.10)

train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=16)
eval_dataloader = DataLoader(dataset2["val"], batch_size=16)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
#model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))
print("start training...")

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        #encoded_input = tokenizer(text, return_tensors='pt')

        #batch1 = {i: tokenizer(batch[0], return_tensors='pt',padding=True).to(device) for i in range(len(batch))}
        #batch2 = {k:tokenizer(j, return_tensors='pt').to(device) for k,j in batch[1]}
        a=batch["a"]
        b=batch["b"]
        #batch1=[tokenizer(k,return_tensors="pt").to(device) for k in batch[0]]
        #batch2=[tokenizer(k,return_tensors="pt").to(device) for k in batch[1]]
        #label=[torch.tensor(k) for k in batch[2]]
        label=batch['label']
        #pred=tokenizer(batch[0][0], return_tensors='pt',padding=True).to(device)
        #outputs1 = [model(k) for k in batch1]
        #outputs2 = [model(k) for k in batch2]
        output_1=model(a)
        output_2=model(b)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_val=[]
        #for k in range(len(outputs2)):
        #    cos_val.append(cos(outputs1[k], outputs2[k]))
        # val=(1+cos_val)/2
        loss=[]
        for k in range(len(label)):
            loss.append(torch.abs(label[k] - cos_val[k]))
        loss=torch.sum(loss)
        loss.backward()
        #outputs2 = model(**batch1)

        #loss = outputs1.loss

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)