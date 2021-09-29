import logging
import math
from datetime import datetime

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample, LoggingHandler
import numpy as np
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator, CEBinaryClassificationEvaluator
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

# Pre-trained cross encoder
#model = CrossEncoder('cross-encoder/stsb-distilroberta-base')

#model = CrossEncoder('distilroberta-base', num_labels=2)
model = CrossEncoder("output/CrossEncoder-training--2021-09-29_01-00-39")

# Read STSb dataset
logger.info("Read  train dataset")
train_batch_size = 32#16
num_epochs=4
model_save_path = 'output/training--'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_samples = []
dev_samples = []
test_samples = []
y=[]
cos_sim=[]
label2int = {"same": 0, "different": 1}




excel_data = pd.read_excel('D:\\PhD\\Notebooks\\captionpairs.xlsx')
data = pd.DataFrame(excel_data, columns=['s1', 's2', 'same_img','id1','id2'])
for index,row in data.iterrows():
    if row["same_img"]:
        y.append(1)
    else:
        y.append(0)
s1_list=data['s1'].to_numpy()
s2_list=data['s2'].to_numpy()
label_list=data['same_img'].to_numpy()
do=lambda x: "same" if x else "different"
label_list=[do(x) for x in label_list]

iter_range=int(len(s1_list)/3)
for i in range(iter_range):
    ex = InputExample(texts=[s1_list[i], s2_list[i]], label=label2int[label_list[i]])
    ex2 = InputExample(texts=[s2_list[i], s1_list[i]], label=label2int[label_list[i]])
    train_samples.append(ex)
    train_samples.append(ex2)
for i in range(iter_range,iter_range*2):
    ex=InputExample(texts=[s1_list[i], s2_list[i]], label=label2int[label_list[i]])
    dev_samples.append(ex)
for i in range(iter_range*2,iter_range*3):
    ex=InputExample(texts=[s1_list[i], s2_list[i]], label=label2int[label_list[i]])
    test_samples.append(ex)

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
# We add an evaluator, which evaluates the performance during training
#evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='dev-Split')
evaluator = CEBinaryClassificationEvaluator.from_input_examples(examples=dev_samples, name="dev-binClass")

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))


#model.fit(train_dataloader=train_dataloader,
          #evaluator=evaluator,
          #epochs=num_epochs,
          #evaluation_steps=5000,
          #warmup_steps=warmup_steps,
          #output_path=model_save_path)
#model.save(model_save_path)


#emb_s1=model.predict(s1_list)
#emb_s2=model.predict(s1_list)
val=[list(a) for a in zip(s1_list,s2_list)]
val=val[iter_range*2:iter_range*3]
#for i in range(len(s1_list)):
#    cos_sim.append(model.predict([s1_list[i],s2_list[i]]))
cos_sim=model.predict(val)
cos_sim=[i[0] for i in cos_sim]

fpr, tpr, thresholds = metrics.roc_curve(y[iter_range*2:iter_range*3], cos_sim)
auc=metrics.auc(fpr, tpr)
print("auc: ",auc)
