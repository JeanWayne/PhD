import json
from os import walk

from krippendorff import krippendorff
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats import inter_rater as irr
from DataClass import *
from Experimental.RankWords import getRankingFromRoberta




def precisionAtN(selected,ranking,n=3):
    1+1

def cohen_kappa_abst(ann1, ann2):
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (P_o)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (P_e)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count

    return round((A - E) / (1 - E), 4)

def fleiss_kappa(annos,all):
    vec=[0]*len(all)
    for ti in range(len(all)):
        tok=all[ti]
        for annotator in annos:
            if tok in annotator:
                vec[ti]+=1
    val=irr.fleiss_kappa(([vec]), method='fleiss')
    return val

def cohen_kappa(a,b,all):
    vec_a=[0]*len(all)
    vec_b=[0]*len(all)
    for t in range(len(all)):
        tok=all[t]
        if tok in a:
            vec_a[t]+=1
        if tok in b:
            vec_b[t]+=1
    score=cohen_kappa_abst(vec_a,vec_b)
    return score,vec_a,vec_b
    #return cohen_kappa_score(vec_a,vec_b)


files= next(walk("completed/"), (None, None, []))[2]
print(files)
datasets=[]
datasets_names=["jean","vitor","christian","josif"]
datasets_agg={0:{},1:{},2:{},3:{}}
count=0
for file in files:
    with open('completed/'+file) as json_file:
        data = json.load(json_file)
        datasets.append(DataSet(data,count))
        count+=1
print(len(datasets))

ITS=[]
ITS_terms={}
ITS_caption={}
with open("unique_IDS.txt") as file:
    for lines in file.readlines():
        ITS.append(lines.replace("\n",""))

for d in datasets:
    for it in d.Items:
        if it['ID'] in ITS:
            try:
                datasets_agg[d.id][it['ID']]=it['selected']
                ITS_terms[it['ID']]=it['tokens']
                ITS_caption[it['ID']]=it['caption']
            except KeyError:
                1+1
print(datasets_agg[0])
print(datasets_agg[1])
print(datasets_agg[2])
print(datasets_agg[3])

for i in range(len(datasets[0].Items)):
    it=datasets[0].Items[i]["ID"]
    if it in ITS:
        print(i,"  :  ",it)


def compareAnnotators(AnoA,AnoB):
    selectA,selectB=AnoA,AnoB
    Aggrement={}
    for t in ITS:
        temp=[]
        temp.append(datasets_agg[selectA][t])
        temp.append(datasets_agg[selectB][t])
        Aggrement[t]=temp

    cohenSum=0
    sumVec_a=[]
    sumVec_b=[]
    fleissSum=0
    rob_rankin=[]


    for t in Aggrement:
        #rob_rankin=getRankingFromRoberta([ITS_caption[t]])
        all_terms=list(set(ITS_terms[t]))
        x=set(Aggrement[t][0])
        y=set(Aggrement[t][1])
        #print(datasets_names[selectA],": ",x)
        #print(datasets_names[selectB],": ",y)
        inter=x.intersection(y)
        #print(len(inter),"    ",inter)
        cohen,va,vb=cohen_kappa(x,y,all_terms)
        sumVec_a.extend(va)
        sumVec_b.extend(vb)
        fleiss=fleiss_kappa([x,y],all_terms)
        #print("roberta top5: ",rob_rankin[:5])
        #print("cohen: ",cohen)
        #print("fleiss: ",fleiss)
        cohenSum+=cohen
        fleissSum+=fleiss
        #print("________________________")
    #print("avg cohen: ",cohenSum/len(ITS))
    #print("all cohen: ",cohen_kappa_abst(sumVec_a,sumVec_b))
    #print("avg fleiss: ",fleissSum/len(ITS))
    return  cohenSum/len(ITS),fleissSum/len(ITS)

for ab in range(len(datasets_names)):
    for ba in range(len(datasets_names)):
        if(ab==ba):
            continue
        cohen,fleiss=compareAnnotators(ab,ba)
        print(datasets_names[ab]," : ",datasets_names[ba],"   ==    COHEN: ","{:9.5f}".format(cohen))#,"        FLEISS: ","{:9.5f}".format(fleiss))



#agg = irr.aggregate_raters(arr)
#agg