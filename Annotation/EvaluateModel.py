import re
import string

import numpy as np
from nltk.corpus import stopwords

from EvaluationFuncs import *
from Experimental.RankWords import getRankingFromRoberta

stoplist = set(stopwords.words('english') + list(string.punctuation) + ["``", "''", "'s"])

def removeDuplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]

def fuseDataSets(ds):
    newDataSet=DataSet()
    newDataSet.id=-1
    for dataset in ds:
        newDataSet.Items.extend(dataset.Items)
    return newDataSet

def replaceUnicodeError(s):
    liste=[("Äį","č"),("Äĩ","ć"),("ÃŃ","í"),("Ã©","é"),("Ãī","É"),("âĢĵ","–"),("Ã¡","á"),("ÈĻ","ș"),("Äĥ","ă"),("Èĺ","Ș"),("Ã³","ó"),("Á","Ãģ"),("Å«","ū"),("Åį","ō"),("ÅĤ","ł"),("ÅĦ","ń"),("ÃŃ","í"),("ÄĲ","Đ")]
    for l in liste:
        s=s.replace(l[0],l[1])
    return s

def RobertFilterTokens(rob,all):
    rob=RobertaFilterString(rob)
    rob=removeDuplicates(rob)
    rob=[(r[0],r[1]) for r in rob if len(r[0]) >1]
    rob=[(r[0],r[1]) for r in rob if r[0].lower() not in stoplist]
    try:
        rob = [(r[0].encode("latin-1").decode("utf-8"),r[1]) if r[0] not in all else (r[0],r[1]) for r in rob]
        #rob=[(r[0].encode("latin-1").decode("utf-8"),r[1]) for r in rob if r[0] not in all]
    except UnicodeEncodeError:
        rob=[(replaceUnicodeError(r[0]),r[1]) for r in rob]

    return rob

def RobertaFilterString(tub):
    punct_list=[a for a in string.punctuation]
    newTub=[]
    for t in tub:
        newString = t[0]
        newString= re.sub(r"\([^()]*\)||\[[^()]*\]", "", newString)
        newString= re.sub(r"\'s", "", newString)
        if len(newString)>0:
            try:
                if newString[0] in punct_list:
                    newString=newString[1:]
                if newString[-1] in punct_list:
                    newString=newString[:-1]
            except IndexError:
                1+1
        newTub.append((newString, t[1]))
    return newTub

#allTokens= ['Kushan', 'king', 'prince', 'said', 'Huvishka', 'Gandhara', 'art']
#selected= ['Kushan', 'Huvishka', 'Gandhara']
#rob= [(',king\'s', 0.27507561099870753), ('Gandhara', 0.253070815917959), ('Huvishka,', 0.23269902746496446), ('art.[1]', 0.2324995781635239), ('prince,', 0.214403399154738), ('or', 0.19999122616302756), ('said', 0.1881358585118764), ('be', 0.14899640279214682), ('to', 0.14664574370447422)]
#print(RobertaFilterString(rob))
#1+1

def getPredictionsForDataset(ds):
    tokenList=[]
    selected=[]
    captionList=[]
    for i in ds.Items:
        try:
            if i['bad'] is not None:
                if i['bad'] is True:
                    continue
            tokenList.append(i['tokens'])
            captionList.append(i['caption'].replace(u'\xa0'," "))
            selected.append(i['selected'])
        except KeyError:
            1+1
    rob_ranking = getRankingFromRoberta(captionList,model_path="../PyTorch/Model/Epoch_1_2022-01-06_18-52-20_AUC_0.8593628700013282") #AUC :  0.843
    #rob_ranking = getRankingFromRoberta(captionList,model_path="../PyTorch/Model/AUC_0.8738789000058799_Epoch_1_2022-03-03_03-48-27") #AUC :  0.634
    #rob_ranking = getRankingFromRoberta(captionList,model_path="../PyTorch/Model/2022-03-03_18-54-05_Epoch_0AUC_0.8850419384902144") #AUC :  0.8100435996348093 MAP :  0.6809963402041768
    #rob_ranking = getRankingFromRoberta(captionList,model_path="../PyTorch/Model/2022-03-03_21-10-42_Epoch_1AUC_0.8778205690477496") #AUC :  0.8017614181397812 MAP :  0.6740292989851439
    #rob_ranking = getRankingFromRoberta(captionList,model_path="../PyTorch/Model/2022-03-03_23-10-33_Epoch_2AUC_0.8683830382106243") #AUC : AUC :  0.7992604759124361 MAP :  0.6755385722097662

    rob_ranking_filtered=[]
    for i in range(len(rob_ranking)):
        try:
            filtered=RobertFilterTokens(rob_ranking[i],tokenList[i])
            filtered.sort(key=lambda tup: tup[1], reverse=True)
            rob_ranking_filtered.append(filtered)
        except:
            print(rob_ranking[i])
            print(tokenList[i])
            print(captionList[i])
            print("____________________")
            #rob_ranking=[RobertFilterTokens(k) for k in rob_ranking]

    return selected,rob_ranking_filtered,tokenList
raw,_=loadDataSets("raw")



datasets,ITS=loadDataSets("completed")

idfs=compute_idf(raw)

at_selector=3
UserList=["jean","vitor","christian","josif"]

newDataSet=fuseDataSets(datasets)
selected, rob, allTokens = getPredictionsForDataset(newDataSet)
AUC_list_rob=[]
MAP_list_rob=[]
prec_list_rob=[]
recall_list_rob=[]

AUC_list_idf=[]
MAP_list_idf=[]
prec_list_idf=[]
recall_list_idf=[]

AUC_list_conc=[]
MAP_list_conc=[]
prec_list_conc=[]
recall_list_conc=[]
for i in range(len(selected)):
    print("allTokens:",allTokens[i])
    tfidf=compute_tf_idf(allTokens[i],idfs)
    conc=compute_concreteness(allTokens[i])
    #rob[i]=tfidf
    print("selected:",selected[i])
    print("rob:",rob[i])
    print("conc:",conc)
    print("idf:", tfidf)
    #acc_r,prec_r,recall_r=metricAtN(selected[i],rob[i],allTokens[i],at_selector)
    #acc_i,prec_i,recall_i=metricAtN(selected[i],tfidf,allTokens[i],at_selector)
    #acc_c,prec_c,recall_c=metricAtN(selected[i],conc,allTokens[i],at_selector)
    prec_r, recall_r,acc_r = evaluate3(allTokens[i],selected[i],rob[i])
    prec_i, recall_i, acc_i = evaluate3(allTokens[i], selected[i], tfidf)
    prec_c, recall_c, acc_c = evaluate3(allTokens[i], selected[i], conc)

    map_s_r=MAP_score(selected[i],rob[i],allTokens[i])
    auc_s_r=auc_score(selected[i],rob[i],allTokens[i])

    map_s_i=MAP_score(selected[i],tfidf,allTokens[i])
    auc_s_i=auc_score(selected[i],tfidf,allTokens[i])

    map_s_c=MAP_score(selected[i],conc,allTokens[i])
    auc_s_c=auc_score(selected[i],conc,allTokens[i])
    print("----------------")
    print("Roberta:")
    print("AUC: ",auc_s_r)
    print("MAP: ",map_s_r)
    print("acc: ",acc_r,"   prec: ",prec_r,"   rec: ",recall_r)
    print("----------------")
    print("tf-idf:")
    print("AUC: ",auc_s_i)
    print("MAP: ",map_s_i)
    print("acc: ",acc_i,"   ",prec_i,"   rec: ",recall_i)
    print("----------------")
    print("Concreteness:")
    print("AUC: ",auc_s_c)
    print("MAP: ",map_s_c)
    print("acc: ",acc_c,"   prec: ",prec_c,"   rec: ",recall_c)
    if i == 44:
        1+1
    AUC_list_rob.append(auc_s_r)
    MAP_list_rob.append(map_s_r)
    prec_list_rob.append(prec_r)
    recall_list_rob.append(recall_r)

    AUC_list_idf.append(auc_s_i)
    MAP_list_idf.append(map_s_i)
    prec_list_idf.append(prec_i)
    recall_list_idf.append(recall_i)

    AUC_list_conc.append(auc_s_c)
    MAP_list_conc.append(map_s_c)
    prec_list_conc.append(prec_c)
    recall_list_conc.append(recall_c)
    print("_________________________________________________________________________________________")
AUC_list_rob=[i for i in AUC_list_rob if not isNaN(i)]
MAP_list_rob=[i for i in MAP_list_rob if not isNaN(i)]
AUC_list_idf=[i for i in AUC_list_idf if not isNaN(i)]
MAP_list_idf=[i for i in MAP_list_idf if not isNaN(i)]
AUC_list_conc=[i for i in AUC_list_conc if not isNaN(i)]
MAP_list_conc=[i for i in MAP_list_conc if not isNaN(i)]

AUC_list_rob=np.asarray(AUC_list_rob)
MAP_list_rob=np.asarray(MAP_list_rob)
prec_list_rob=np.asarray(prec_list_rob)
recall_list_rob=np.asarray(recall_list_rob)

AUC_list_idf=np.asarray(AUC_list_idf)
MAP_list_idf=np.asarray(MAP_list_idf)
prec_list_idf=np.asarray(prec_list_idf)
recall_list_idf=np.asarray(recall_list_idf)

AUC_list_conc=np.asarray(AUC_list_conc)
MAP_list_conc=np.asarray(MAP_list_conc)
prec_list_conc=np.asarray(prec_list_conc)
recall_list_conc=np.asarray(recall_list_conc)
print("===========================================================================================")
print("ROBERTA:")
print("AUC : ",np.mean(AUC_list_rob))
print("MAP : ",np.mean(MAP_list_rob))
print("Prec@",at_selector," : ",np.mean(prec_list_rob))
print("Recall@",at_selector," : ",np.mean(recall_list_rob))
print("===========================================================================================")
print("IDF:")
print("AUC : ",np.mean(AUC_list_idf))
print("MAP : ",np.mean(MAP_list_idf))
print("Prec@",at_selector," : ",np.mean(prec_list_idf))
print("Recall@",at_selector," : ",np.mean(recall_list_idf))
print("===========================================================================================")
print("CONCR:")
print("AUC : ",np.mean(AUC_list_conc))
print("MAP : ",np.mean(MAP_list_conc))
print("Prec@",at_selector," : ",np.mean(prec_list_conc))
print("Recall@",at_selector," : ",np.mean(recall_list_conc))

# for dataset in datasets:
#     selected,rob,allTokens=getPredictionsForDataset(dataset)
#     AUC_list=[]
#     MAP_list=[]
#     prec_list=[]
#     recall_list=[]
#     for i in range(len(selected)):
#         tfidf=compute_tf_idf(allTokens[i],idfs)
#         conc=compute_concreteness(allTokens[i])
#         rob[i]=tfidf
#         print("selected:",selected[i])
#         print("rob:",rob[i])
#         print("conc:",conc)
#         print("idf:", tfidf)
#         acc,prec,recall=metricAtN(selected[i],rob[i],allTokens[i],at_selector)
#         map_s=MAP_score(selected[i],rob[i])
#         auc_s=auc_score(selected[i],rob[i])
#         print("AUC: ",auc_s)
#         print("MAP: ",map_s)
#         print(acc,"   ",prec,"   ",recall)
#         if i == 44:
#             1+1
#         AUC_list.append(auc_s)
#         MAP_list.append(map_s)
#         prec_list.append(prec)
#         recall_list.append(recall)
#
#         #print("_________________________________________________________________________________________")
#     AUC_list=[i for i in AUC_list if not isNaN(i)]
#     MAP_list=[i for i in MAP_list if not isNaN(i)]
#     AUC_list=np.asarray(AUC_list)
#     MAP_list=np.asarray(MAP_list)
#     prec_list=np.asarray(prec_list)
#     recall_list=np.asarray(recall_list)
#     print(UserList[dataset.id]," - Means:")
#     print("AUC : ",np.mean(AUC_list))
#     print("MAP : ",np.mean(MAP_list))
#     print("Prec@",at_selector," : ",np.mean(prec_list))
#     print("Recall@",at_selector," : ",np.mean(recall_list))
