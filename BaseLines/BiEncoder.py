import logging
import math
from datetime import datetime
from random import random

from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator
from sklearn.utils import shuffle

import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample, LoggingHandler, losses, SentenceTransformer, models
import numpy as np
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator, CEBinaryClassificationEvaluator
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader

print("Start")
#### Just some code to print debug information to stdout
def bool2int(b):
    if b:
        return 0.95
    else:
        return 0.05

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
# Pre-trained cross encoder
model_name = 'roberta-base'#sentence-transformers/distiluse-base-multilingual-cased-v2'#'roberta-base'#'bert-base-uncased'
word_embedding_model = models.Transformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode = 'cls', pooling_mode_cls_token = True)

bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])
#model_name= 'output/bi-encoder/sentence-transformers-distiluse-base-multilingual-cased-v2-2021-10-11_16-31-13'
max_seq_length = 128
logging.info("Loading bi-encoder model: {}".format(model_name))

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
#word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)

# Apply mean pooling to get one fixed sized sentence vector
#pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())#,pooling_mode="cls")

#bi_encoder = SentenceTransformer(modules=[word_embedding_model, pooling_model])#SentenceTransformer("output/bi-encoder/bert-base-uncased-2021-09-30_00-19-38") #
#bi_encoder=SentenceTransformer(model_name)

# Read STSb dataset
logger.info("Read  train dataset")
train_batch_size = 8#16
num_epochs=1
bi_encoder_path = 'output/bi-encoder/'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_samples = []
dev_samples = []
test_samples = []
y=[]
cos_sim=[]
label_list=[]



excel_data = pd.read_excel('D:\\PhD\\Notebooks\\captionpairs.xlsx')
df = pd.DataFrame(excel_data, columns=['s1', 's2', 'same_img','id1','id2'])
#data = shuffle(data)
data = list(zip(df.s1,df.s2,df.same_img))

import random

random.shuffle(data)

split = (len(data)//10)
train = data[split*2:]
dev = data[split:split*2]
test = data[:split]
train_examples = [InputExample(texts=[a,b], label=bool2int(v)) for a,b,v in train]
train_examples.extend([InputExample(texts=[b,a], label=bool2int(v)) for a,b,v in train])
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=train_batch_size)
dev_examples = [InputExample(texts=[a,b],label=bool2int(v)) for a,b,v in dev]
train_loss = losses.CosineSimilarityLoss(model=bi_encoder)
print("Train Size: ",str(len(train_examples)),"  Test Size: ",str(len(test)))

logging.info("Read  dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_examples, name='dev')
#evaluator = BinaryClassificationEvaluator.from_input_examples(dev_examples,name='dev')
# Configure the training.
warmup_steps = 100 #math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the bi-encoder model
bi_encoder.fit(train_objectives=[(train_dataloader, train_loss)],
          #evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=bi_encoder_path
          )

#bi_encoder.save(bi_encoder_path)


embeddings_a = bi_encoder.encode([a for a,b,_ in test])
embeddings_b = bi_encoder.encode([b for a,b,_ in test])


def bi_encoder_cos(n):
    e1 = embeddings_a[n]
    e2 = embeddings_b[n]
    sim = cosine_similarity([e1],[e2])
    return sim[0][0]

truevals = [v for _,_,v in test]
predictions = [bi_encoder_cos(n) for n in range(len(test))]

#cos_sim=bi_encoder.predict(val)
#cos_sim=[i[0] for i in cos_sim]

print("Train Size: ",str(len(train_examples)),"  Test Size: ",str(len(test)))
fpr, tpr, thresholds = metrics.roc_curve(truevals, predictions, pos_label=True)
auc=metrics.auc(fpr, tpr)
print("auc: ",auc)
