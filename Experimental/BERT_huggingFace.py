from transformers import pipeline;
sent="Moritz is an evil bastard. He wanted to destroy the world."
print(sent)
print(pipeline('sentiment-analysis')(sent))