import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from transformers import RobertaModel, RobertaTokenizer, DistilBertTokenizer, DistilBertModel
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

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

def fuseTokens(tokens):
    newList=["<s>"]
    IndexList=[0]
    wordCount=1
    for i in range(1,len(tokens)):
        if tokens[i] in ["</s>"] or tokens[i][0]=="Ġ":
            wordCount+=1
        IndexList.append(wordCount)
    numOfWords=IndexList[-2:][0]
    for i in range(1,numOfWords+1):
        first_pos = IndexList.index(i)
        last_pos = len(IndexList) - IndexList[::-1].index(i) - 1
        #print(tokens[first_pos], tokens[last_pos])
        if first_pos!=last_pos:
            newList.append(''.join(tokens[first_pos:last_pos+1]))
        else:
            newList.append(tokens[first_pos])
    newList.append("</s>")

    return newList,IndexList

def fuseAttention(tokens,att):
    newTokens, indexList = fuseTokens(tokens)
    numOfWords=indexList[-1]+1
    attention=[0]*numOfWords
    att=att[0]
    for i in range(att.size):
        attention[indexList[i]]+=att[i]
    #normalize:
    for i in range(len(attention)):
        attention[i]=attention[i]/indexList.count(i)
    newerTokens=[]
    for t in newTokens:
        if t[0]=="Ġ":
            newerTokens.append(t[1:])
        else:
            newerTokens.append(t)
    return newerTokens,attention

def getLastTokenAtt(g):
    return np.array([[s[-1:][0] for s in g]])

def getRankingFromRoberta(inputList,aggregate_layers=5,sort_result=True,remove_EOS=True,model_path="../PyTorch/Model/Epoch_2_2022-01-06_20-39-28_AUC_0.8567258467979885"):
    model=RobertaModel.from_pretrained(model_path, output_attentions=True)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    kk=0
    returnList=[]
    for s in inputList:
        kk += 1
        sentence_a = s
        inputs = tokenizer.encode_plus(sentence_a, return_tensors='pt', add_special_tokens=True)
        input_ids = inputs['input_ids']
        attention = model(input_ids)[-1]
        input_id_list = input_ids[0].tolist()  # Batch index 0
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)
        attn_data = []
        sentence_b_start = None
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

        for k in [aggregate_layers]:
            dataFrames=[]
            for j in range(k,12):
                for i in range(12):
                    df = np.asarray(attn_data[0]['attn'][j][i])/((12-k)*12)
                    dataFrames.append(df)
            dataFrames=sum(dataFrames)
            lastLayer=getLastTokenAtt(dataFrames)
            newTokens, newAtt = fuseAttention(tokens, lastLayer)
            ergebnis_liste=list(zip(newTokens, newAtt))
            if sort_result:
                ergebnis_liste.sort(key=lambda tup: tup[1],reverse=True)
            if remove_EOS:
                ergebnis_liste = [i for i in ergebnis_liste if i[0] not in ['</s>','<s>']]

            returnList.append(ergebnis_liste)
    return returnList
#a=getRankingFromRoberta(['President Obama delivered the 2011 State of the Union Address on January 25, 2011'])
#print(a)

