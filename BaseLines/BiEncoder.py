import logging
import math
from datetime import datetime

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sklearn.utils import shuffle

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample, LoggingHandler, losses, SentenceTransformer, models
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
model_name = 'bert-base-uncased'
max_seq_length = 128
logging.info("Loading bi-encoder model: {}".format(model_name))

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

bi_encoder = SentenceTransformer("output/bi-encoder/bert-base-uncased-2021-09-30_00-19-38") #SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Read STSb dataset
logger.info("Read  train dataset")
train_batch_size = 32#16
num_epochs=4
bi_encoder_path = 'output/bi-encoder/'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_samples = []
dev_samples = []
test_samples = []
y=[]
cos_sim=[]
label2int = {"same": -1., "different": 1.}




excel_data = pd.read_excel('D:\\PhD\\Notebooks\\captionpairs.xlsx')
data = pd.DataFrame(excel_data, columns=['s1', 's2', 'same_img','id1','id2'])
data = shuffle(data)

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



train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=bi_encoder)

logging.info("Read  dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the bi-encoder model
#bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
#          evaluator=evaluator,
#          epochs=num_epochs,
#          evaluation_steps=1000,
#          warmup_steps=warmup_steps,
#          output_path=bi_encoder_path
#          )
#bi_encoder.save(model_save_path)


#emb_s1=model.predict(s1_list)
#emb_s2=model.predict(s1_list)
val=[list(a) for a in zip(s1_list,s2_list)]
val=val[iter_range*2:iter_range*3]
a_vec=[a[0] for a in val]
b_vec=[a[0] for a in val]
a_vec=bi_encoder.encode(a_vec)
b_vec=bi_encoder.encode(b_vec)

for i in range(len(a_vec)):
    cos = cosine_similarity(a_vec[i].reshape(1, -1), b_vec[i].reshape(1, -1))
    cos_sim.append(cos[0][0])
#cos_sim=bi_encoder.predict(val)
#cos_sim=[i[0] for i in cos_sim]

fpr, tpr, thresholds = metrics.roc_curve(y[iter_range*2:iter_range*3], cos_sim)
auc=metrics.auc(fpr, tpr)
print("auc: ",auc)
