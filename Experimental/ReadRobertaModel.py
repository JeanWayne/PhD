import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from transformers import RobertaModel, RobertaTokenizer, DistilBertTokenizer, DistilBertModel
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()

def format_special_chars(tokens):
    return [t.replace('Ġ', ' ').replace('▁', ' ').replace('</w>', '') for t in tokens]

def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

model_version = 'roberta-base'
#model_path = "../BaseLines/output/CrossEncoder-training--2021-09-29_21-55-46"
#model_path = "../BaseLines/output/bi-encoder/roberta-base-2021-10-12_09-28-49"
model_path ="../BaseLines/output/bi-encoder/roberta-base-2021-10-13_09-54-56"
model = RobertaModel.from_pretrained(model_path, output_attentions=True)
tokenizer = RobertaTokenizer.from_pretrained(model_path)
sentence_a = "President Obama delivered the 2011 State of the Union Address on January 25, 2011"
#sentence_b = "The cat lay on the rug"
inputs2 = tokenizer.encode(sentence_a,return_tensors='pt',add_special_tokens=True)
inputs = tokenizer.encode_plus(sentence_a, return_tensors='pt', add_special_tokens=True)
input_ids = inputs['input_ids']
attention = model(input_ids)[-1]
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)
print(input_ids)
print(tokens)
x=format_attention(attention)
print("done")

attn_data = []

sentence_b_start = None
prettify_tokens = True
layer = None
heads = None
encoder_attention = None
decoder_attention = None
cross_attention = None
encoder_tokens = None
decoder_tokens = None
include_layers = None
if attention is not None:
    if include_layers is None:
        include_layers = list(range(len(attention)))
    attention = format_attention(attention, include_layers)
    if sentence_b_start is None:
        attn_data.append(
            {
                'name': None,
                'attn': attention.tolist(),
                'left_text': tokens,
                'right_text': tokens
            }
        )
    else:
        slice_a = slice(0, sentence_b_start)  # Positions corresponding to sentence A in input
        slice_b = slice(sentence_b_start, len(tokens))  # Position corresponding to sentence B in input
        attn_data.append(
            {
                'name': 'All',
                'attn': attention.tolist(),
                'left_text': tokens,
                'right_text': tokens
            }
        )
        attn_data.append(
            {
                'name': 'Sentence A -> Sentence A',
                'attn': attention[:, :, slice_a, slice_a].tolist(),
                'left_text': tokens[slice_a],
                'right_text': tokens[slice_a]
            }
        )
        attn_data.append(
            {
                'name': 'Sentence B -> Sentence B',
                'attn': attention[:, :, slice_b, slice_b].tolist(),
                'left_text': tokens[slice_b],
                'right_text': tokens[slice_b]
            }
        )
        attn_data.append(
            {
                'name': 'Sentence A -> Sentence B',
                'attn': attention[:, :, slice_a, slice_b].tolist(),
                'left_text': tokens[slice_a],
                'right_text': tokens[slice_b]
            }
        )
        attn_data.append(
            {
                'name': 'Sentence B -> Sentence A',
                'attn': attention[:, :, slice_b, slice_a].tolist(),
                'left_text': tokens[slice_b],
                'right_text': tokens[slice_a]
            }
        )

sent=np.zeros(len(attn_data[0]['attn'][0][0]))
for d in attn_data:
    for layer in d['attn']:
        for head in layer:
            for word in range(len(head)):
                1+1
                #sent[word]+=head[word]
print(sent)
layerNum=0
fig = plt.figure()
fig.tight_layout()
fig.subplots_adjust(top=0.95,bottom=0.1)
heads = np.asarray(attn_data[0]['attn'][0][1])
ax = sns.heatmap(heads,linewidths=.5)
ax.set_xticklabels(tokens,rotation=90)
ax.set_yticklabels(tokens,rotation=0)
plt.show()

fig, axn = plt.subplots(3, 4, sharex=True, sharey=True)
fig.set_size_inches(18.5, 10.5, forward=True)
cbar_ax = fig.add_axes([.91, .3, .03, .4])
for i, ax in enumerate(axn.flat):
    df = pd.DataFrame(np.asarray(attn_data[0]['attn'][11][i]))
    sns.heatmap(df,ax=ax,linewidths=.5,xticklabels="auto",yticklabels="auto")
    #ax.set_xticklabels(tokens,rotation=90)
    #ax.set_yticklabels(tokens,rotation=0)
fig.tight_layout(rect=[0, 0, .9, 1])

plt.show()


