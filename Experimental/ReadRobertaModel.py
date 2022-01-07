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


model_version = 'roberta-base'
#model_path = "../BaseLines/output/CrossEncoder-training--2021-09-29_21-55-46"
#model_path = "../BaseLines/output/bi-encoder/roberta-base-2021-10-12_09-28-49"
#model_path ="../BaseLines/output/bi-encoder/roberta-base-2021-10-13_09-54-56" #CLS token
#model_path= "../BaseLines/output/bi-encoder/output-bi-encoder-roberta-base-2021-10-13_09-54-56-2021-11-07_14-39-37"#no cls token
#model_path = "../BaseLines/output/CrossEncoder-training--2021-11-07_15-43-27" #Crossencoder
model_path ="../PyTorch/Model/Epoch_2_2022-01-06_20-39-28_AUC_0.8567258467979885"
model = RobertaModel.from_pretrained(model_path, output_attentions=True)
tokenizer = RobertaTokenizer.from_pretrained(model_path)
sentence_a = "President Obama delivered the 2011 State of the Union Address on January 25, 2011"
#sentence_a="Members of Trotski's Left Opposition, 1927. Smirnov is the second to the left, seated next to Trotski"
#sentence_a="The University clocktower, looking east."
#sentence_a="Haggis, neeps and tatties"
#sentence_a="Ben Bella and Fidel Castro Meeting in Cuba"
#sentence_a="A visual form of recursion known as the Droste effect. The woman in this image is holding an object which contains a smaller image of her holding the same object, which in turn contains a smaller image of herself holding the same object, and so forth."
#sentence_a="Autonomous and remote controlled drones are now a common sight in a war zone"
#sentence_a="This female Great Frigatebird has been tagged with wing tags as part of a breeding study"
#sentence_a="gfffddasdf 23asdf wef9a Castle castle"
#sentence_a="Mosaic in Byzantine style, Palermo, 1150"

#sentence_a="French torpedo launch attacking the Chinese frigate Yuyuan, 14 February 1885"
#sentence_a="French Navy torpedo boat attacking Yuyuen."
#sentence_b = "The cat lay on the rug"
inputList=["President Obama delivered the 2011 State of the Union Address on January 25, 2011",
           "Members of Trotski's Left Opposition, 1927. Smirnov is the second to the left, seated next to Trotski",
           "The University clocktower, looking east.",
           "Ben Bella and Fidel Castro Meeting in Cuba",
           "Autonomous and remote controlled drones are now a common sight in a war zone",
           "This female Great Frigatebird has been tagged with wing tags as part of a breeding study",
           "Mosaic in Byzantine style, Palermo, 1150",
           "French torpedo launch attacking the Chinese frigate Yuyuan, 14 February 1885",
           "French Navy torpedo boat attacking Yuyuen."]
           #"French torpedo launch attacking the Chinese frigate Yuyuan, 14 February 1885. French Navy torpedo boat attacking Yuyuen."]

inputList=["President Obama delivered the 2011 State of the Union Address on January 25, 2011"]
kk=0
for s in inputList:
    kk+=1
    sentence_a=s
    inputs2 = tokenizer.encode(sentence_a,return_tensors='pt',add_special_tokens=True)
    inputs = tokenizer.encode_plus(sentence_a, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    attention = model(input_ids)[-1]
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)



    #print(input_ids)
    print(tokens)
    print(fuseTokens((tokens)))
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
    heads = np.asarray(attn_data[0]['attn'][11][1])
    #ax = sns.heatmap(heads,linewidths=.5)
    #ax.set_xticklabels(tokens,rotation=90)
    #ax.set_yticklabels(tokens,rotation=0)
    #plt.show()
    sns.set(rc={'figure.figsize':(18.7,15.27),"font.size":12,"axes.titlesize":30,"axes.labelsize":20})
    sns.color_palette("coolwarm", as_cmap=True)

    def colors_from_values(values, palette_name):
        # normalize the values to range [0, 1]
        normalized = (values - min(values)) / (max(values) - min(values))
        # convert to indices
        indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
        # use the indices to get the colors
        palette = sns.color_palette(palette_name, len(values))
        return np.array(palette).take(indices, axis=0)

    def getLastTokenAtt(g):
        return np.array([[s[-1:][0] for s in g]])
    for k in [0,5,8,11]:#range(12):
        #fig, axn = plt.subplots(3, 4, sharex=True, sharey=True)
        fig.set_size_inches(24.5, 18.5, forward=True)
        dataFrames=[]
        for j in range(k,12):
            for i in range(12):
                df = np.asarray(attn_data[0]['attn'][j][i])/((12-k)*12)
                dataFrames.append(df)
        #dataFrames=dataFrames[10:]
        dataFrames=sum(dataFrames)
        lastLayer=getLastTokenAtt(dataFrames)
        newTokens, newAtt = fuseAttention(tokens, lastLayer)
        lastLayer_df=pd.DataFrame(data=dataFrames[0:,0:],
                    index=[i for i in range(dataFrames.shape[0])],
                    columns=tokens)
        dataframes=pd.DataFrame(dataFrames)

        ax = sns.heatmap(dataframes,linewidths=.5, annot = True)
        pal = sns.color_palette("Greens_d", len(lastLayer[0]))
        rank = lastLayer[0].argsort().argsort()
        #sns.barplot(y=tokens, x=lastLayer[0]*len(tokens),palette=np.array(pal[::-1])[rank])
        path_for_img="D:/PhD/images/attention/"
        plt.savefig(path_for_img+str(kk)+newTokens[1]+"-"+str(k+1)+'-12_crossEncoder_HeatMap.png', bbox_inches='tight')
        plt.show()

        ax = sns.barplot(x=newTokens, y=newAtt,palette=colors_from_values(newAtt, "YlOrRd"))

        ax.set_title('Layer '+str(k+1)+' - 12 for all Attention Heads\n For Sentence:\n'+sentence_a)
        for item in ax.get_xticklabels():
            item.set_rotation(45)
        ax.tick_params(labelsize=25)
        #ax.set_xticklabels(tokens,rotation=90)
        #ax.set_yticklabels(tokens,rotation=0)
        plt.savefig(path_for_img+str(kk)+newTokens[1]+"-"+str(k+1)+'-12_crossEncoder_LastColumn.png', bbox_inches='tight')

        plt.show()


