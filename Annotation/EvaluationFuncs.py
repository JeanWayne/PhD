import json
import math
from collections import Counter
from os import walk

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_curve, auc, average_precision_score

from Annotation.DataClass import DataSet
from Concreteness.concreteness_prediction import getConcreteness

import numpy as np



################################################################################################################
globaleval = []


def wc_ap(selected, results):
    sigma = 0
    tp = 0
    for n in range(len(results)):
        if results[n] in selected:
            tp += 1
            sigma += tp / (n + 1)
    return sigma / len(selected)


def wc_auc(selected, results):
    if len(results) == len(selected):
        return 1
    area = 0
    width = 1 / (len(results) - len(selected))
    tp = 0
    for t in results:
        if t in selected:
            tp += 1
        else:
            area += ((tp + 0.5) / len(selected)) * width

    return area


def evaluate3(terms, selected, results):
    # print(selected)

    result=[s for (s,_) in results]
    tp = len([t for t in result[:3] if t in selected])
    tn = len([t for t in result[3:] if t not in selected])
    # fp = len([t for t in result if t not in selected])
    precision = tp / 3
    recall = tp / len(selected)
    accuracy = (tp + tn) / len(terms)
    return precision, recall, accuracy

#########################################################################################################################

def isNaN(num):
    return num != num

def metricAtN(true,pred,all,n):

    #selected = [t for t in selected if t in terms]
    #roberta = [t for t in roberta if t in terms] + [t for t in terms if t not in roberta]
    pred=pred[:n]
    pred=[p[0] for p in pred]
    return metric(true,pred,all)

def metric(true,pred,all):
    #print(true,pred,all)
    #true = [t for t in true if t in all]
    #pred = [t for t in pred if t in all] + [t for t in all if t not in pred]
    #print(true,pred,all)

    trueNeg=len([k for k in all if k not in true and k not in pred])
    truePos=len([k for k in pred if k in true])
    falsePos=len([k for k in pred if k not in true])
    falseNeg=len([k for k in true if k not in pred])

    try:
        acc=(truePos+trueNeg)/(truePos+trueNeg+falsePos+falseNeg)
        prec=truePos/(truePos+falsePos)
        recall=truePos/(truePos+falseNeg)
    except ZeroDivisionError:
        print("Zero div error!")
    if "Drumtochty" in all:
        print("!")
    return acc,prec,recall

def auc_score(true,pred,all):
    all=list(set(all))
    y_true=[]
    scores=[]
    dic_pred=dict(pred)
    for a in all:
        y_true.append(1) if a in true else y_true.append(0)
        try:
            scores.append(dic_pred[a])
        except:
            scores.append(0)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    return auc(fpr, tpr)


# return wc_auc(true,pred)
    #y_true = [1 if i[0] in true else 0 for i in pred]
   # scores=[i[1] for i in pred]
# if isinstance(pred[0], (list, tuple)) and  not isinstance(pred[0], str):
#    pred=[i[0] for i in pred]
#    pred= [t for t in pred if t in all] + [t for t in all if t not in pred]


def MAP_score(true,pred,all):
    all=list(set(all))
    y_true=[]
    scores=[]
    dic_pred=dict(pred)
    for a in all:
        y_true.append(1) if a in true else y_true.append(0)
        try:
            scores.append(dic_pred[a])
        except:
            scores.append(0)
    return average_precision_score(y_true, scores)

def compute_tf(wordDic,bow):
    tfDic={}
    bowCount=len(bow)
    for w, count in wordDic.items():
        tfDic[w] = count/float(bowCount)
    return tfDic

def get_documentBows(datasets):
    docs=[]
    counter=[]
    for k in datasets:
        for item in k.Items:
            counter.extend(item["tokens"])
            docs.append(item['tokens'])
    return docs,Counter(counter)


def compute_idf(datasets):
    corpus,most_freq=get_documentBows(datasets)
    word_idf_values = {}
    for token in most_freq:
        doc_containing_word = 0
        for document in corpus:
            if token in document:
                doc_containing_word += 1
        word_idf_values[token] = np.log(len(corpus) / (1 + doc_containing_word))
    sort_orders = sorted(word_idf_values.items(), key=lambda x: x[1])
    return word_idf_values
def compute_tf_idf(tokens,idfs):
    tfidf=[]
    c= Counter(tokens)
    for t in tokens:
        tfidf.append((t,c[t]*idfs[t]))
    tfidf =sorted(tfidf, key=lambda x: x[1],reverse=True)
    return tfidf

def compute_concreteness(tokens):
    concList=[(t,getConcreteness(t)) for t in tokens]
    concList=sorted(concList,key=lambda x: x[1],reverse=True)
    return concList

def loadDataSets(folder):
    files= next(walk(folder+"/"), (None, None, []))[2]
    print(files)
    datasets=[]
    datasets_names=["jean","vitor","christian"]
    datasets_agg={0:{},1:{},2:{}}
    count=0
    for file in files:
        with open(folder+'/'+file) as json_file:
            data = json.load(json_file)
            datasets.append(DataSet(data,count))
            count+=1
    print("Datasets loaded: ",len(datasets))
    ITS=[]

    with open("unique_IDS.txt") as file:
        for lines in file.readlines():
            ITS.append(lines.replace("\n",""))
    return datasets,ITS

###delete this!
allTokens= ['Istanbul', 'Pride', 'Taksim', 'Square', '2013']
selected= ['Pride', 'Istanbul']
rob= [('Pride', 0.333383806504398), ('Istanbul', 0.2890511695829142), ('Taksim', 0.2749504624559796), ('2013', 0.2708118375227232), ('Square)', 0.26601326680923065)]
conc= [('Square', [3.96622914]), ('Istanbul', [3.72079288]), ('Taksim', [3.57496288]), ('Pride', [3.36702963]), ('2013', [3.0402273])]
idf= [('Pride', 7.313220387090301), ('Taksim', 7.313220387090301), ('Istanbul', 6.620073206530356), ('Square', 5.521460917862246), ('2013', 4.748271029628765)]
print(evaluate3(allTokens, selected, rob))
print(evaluate3(allTokens, selected, idf))
print(evaluate3(allTokens, selected, conc))
print("AUC")
print(auc_score(selected,rob,allTokens))
print(auc_score(selected,idf,allTokens))
print(auc_score(selected,conc,allTokens))
print("AP")
print(MAP_score(selected,idf,allTokens))
print(MAP_score(selected,idf,allTokens))
print(MAP_score(selected,idf,allTokens))
#print("Rob auc: ",auc_score(selected,rob,allTokens))
#print("Rob AP: ",MAP_score(selected,rob,allTokens))
#print("conc auc: ",auc_score(selected,conc,allTokens))
#print("conc AP: ",MAP_score(selected,conc,allTokens))
#print("IDF auc: ",auc_score(selected,idf,allTokens))
#print("IDF AP: ",MAP_score(selected,idf,allTokens))