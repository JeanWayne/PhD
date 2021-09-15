from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
from MongoDB.Query import getNCaptions

model = SentenceTransformer('all-mpnet-base-v2')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    #print("Embedding:", embedding)
    print("")

#for s in embeddings:
#    for k in embeddings:
#        print(cosine_similarity(s.reshape(1, -1),k.reshape(1, -1))[0])

label,caption=getNCaptions(4)
embs= model.encode(caption)
for l in range(len(embs)):
    cos=cosine_similarity(embs[l].reshape(1,-1),embs[11].reshape(1,-1))
    print(label[l]," : ",cos[0])
