
import json
from DataClass import DataSet,Item
from collections import Counter
from os import walk

from Experimental.RankWords import getRankingFromRoberta


def loadFile(filePath):
    datasets = []
    count = 0
    with open("D:\PhD\Annotator\Annotator-josif-test\Annotator\Label_File_#1.json") as json_file:
        data = json.load(json_file)
        datasets.append(DataSet(data,count))
        count+=1
    return datasets

def getData_from_DataSet(ds):
    selected=[]
    captions=[]
    for i in ds[0].Items:
        selected.append(i['selected'])
        captions.append(i['caption'])
    return selected,captions

def compareResults(selected,rankings):
    rankings=rankings[:25]
    selected=selected[:25]
    for f in range(len(rankings)):
        res=rankings[f][:3]
        sel=selected[f]
        print("Top3:  ",res)
        print("Josif: ",sel)
        print("####################################################################################################")



d = loadFile("")
selected,captions=getData_from_DataSet(d)
results=[""]
results=getRankingFromRoberta(captions)
compareResults(selected,results)
