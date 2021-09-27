from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('all-mpnet-base-v2')

#Sentences are encoded by calling model.encode()
dataset=[]
with open("D:\\PhD\\abbr\\abb_1520941690603.txt",encoding="utf-8",mode="r") as file:
    for line in file.readlines():
        dataset.append(line.split("\t"))
print(dataset[:2])