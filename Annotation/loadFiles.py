import json
from collections import Counter
from os import walk

class DataSet():
    def __init__(self,json,id):
        self.Items=[]
        self.id=id
        self.processJSON(json)
    def processJSON(self,json):
        for k in json:
            item={
                "ID":k['ID'],
                "caption":k['caption'],
                "tokens":k['tokens'],
                "URL":k['URL'],
                "selected":k['selected']}
            try:
                item["unsure"]=k['unsure']
            except:
                item["unsure"]=None
            self.Items.append(item)

            #self.Items.append(Item(k['id'],k['caption'],k['tokens'],k['url'],k['selected']))


class Item():
    def __init__(self,id,caption,tokens,url,selected):
        self.ID=id
        self.caption=caption
        self.tokens=tokens
        self.URL=url
        self.selected=[]



files= next(walk("Files/"), (None, None, []))[2]
print(files)
datasets=[]
count=0
for file in files:
    with open('Files/'+file) as json_file:
        data = json.load(json_file)
        datasets.append(DataSet(data,count))
        count+=1
print(len(datasets))
count=0
found=[]
map={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
ITS=['5f899af2bc752fce7ba90cd4', '5f8999e9bc752fce7ba8222a', '5f899a53bc752fce7ba8a5fc', '5f899bb1bc752fce7ba964b6', '5f899bcdbc752fce7ba970ac', '5f8999fbbc752fce7ba849ab', '5f899a39bc752fce7ba88f5f', '5f899a3ebc752fce7ba89463', '5f899ad0bc752fce7ba8f9c0', '5f899c38bc752fce7ba99b0f', '5f899b9cbc752fce7ba95bfa', '5f899aa1bc752fce7ba8ddbc', '5f899abdbc752fce7ba8eeb5', '5f899a0bbc752fce7ba86133', '5f899c28bc752fce7ba994c4', '5f899bc2bc752fce7ba96c15', '5f899d32bc752fce7ba9f220', '5f899d1bbc752fce7ba9ea8f', '5f899a3abc752fce7ba890eb', '5f8999ffbc752fce7ba85015', '5f899b53bc752fce7ba93c20', '5f899cc9bc752fce7ba9cf42', '5f899b1ebc752fce7ba92339', '5f8999e6bc752fce7ba81527', '5f899b96bc752fce7ba959cc', '5f899b99bc752fce7ba95ad1', '5f899aaebc752fce7ba8e615', '5f899a02bc752fce7ba85541', '5f899acfbc752fce7ba8f93b', '5f899dd3bc752fce7baa2379', '5f899a9dbc752fce7ba8daf1', '5f899a6ebc752fce7ba8ba03', '5f899a3fbc752fce7ba89544', '5f899cd0bc752fce7ba9d159', '5f899debbc752fce7baa2ae8', '5f899ad7bc752fce7ba8fd88', '5f8999f0bc752fce7ba836d3', '5f899ae0bc752fce7ba902fc', '5f899af5bc752fce7ba90e40', '5f899a2bbc752fce7ba8828a', '5f899d76bc752fce7baa0772', '5f899cf1bc752fce7ba9dcd8', '5f899c53bc752fce7ba9a4b5', '5f899cedbc752fce7ba9db63', '5f8999f1bc752fce7ba83818', '5f899c52bc752fce7ba9a4a4', '5f899c32bc752fce7ba998cf', '5f899a7dbc752fce7ba8c496', '5f899aecbc752fce7ba909a1', '5f899a2cbc752fce7ba883c2', '5f899b5ebc752fce7ba9412d', '5f899d8bbc752fce7baa0de5', '5f899b30bc752fce7ba92c2c', '5f899c2cbc752fce7ba9968d', '5f899c0cbc752fce7ba98a22', '5f899a1cbc752fce7ba8741e', '5f899dd9bc752fce7baa254e', '5f899addbc752fce7ba900e5', '5f899bb8bc752fce7ba967f6', '5f899d78bc752fce7baa0811', '5f8999e8bc752fce7ba820a4', '5f899de9bc752fce7baa2a5d', '5f899a1abc752fce7ba871df', '5f899bdebc752fce7ba977ab', '5f899cf8bc752fce7ba9def5', '5f899a6fbc752fce7ba8bad2', '5f899d00bc752fce7ba9e1a5', '5f899af3bc752fce7ba90d23', '5f899a68bc752fce7ba8b616', '5f899a74bc752fce7ba8be4d', '5f899b0bbc752fce7ba919b8', '5f899d17bc752fce7ba9e96a', '5f899aa4bc752fce7ba8df6b', '5f899bcebc752fce7ba9710e', '5f899a99bc752fce7ba8d8bf', '5f899acabc752fce7ba8f66a', '5f899abcbc752fce7ba8ee21', '5f899a65bc752fce7ba8b42d', '5f8999f3bc752fce7ba83c45', '5f899b2cbc752fce7ba92a41', '5f899d0fbc752fce7ba9e69c', '5f8999efbc752fce7ba8336b', '5f899da0bc752fce7baa1431', '5f899d72bc752fce7baa0654', '5f899a8cbc752fce7ba8cffe', '5f899b1dbc752fce7ba922ac', '5f899a0bbc752fce7ba860f5', '5f899a91bc752fce7ba8d33b', '5f899ae1bc752fce7ba9032e', '5f8999e9bc752fce7ba823fc', '5f899d76bc752fce7baa0792', '5f899cb5bc752fce7ba9c813', '5f8999f5bc752fce7ba83f4d', '5f899c0cbc752fce7ba989fc', '5f899b80bc752fce7ba950a5', '5f899aacbc752fce7ba8e484', '5f899d27bc752fce7ba9ee96', '5f8999e8bc752fce7ba82124', '5f899a64bc752fce7ba8b300', '5f899defbc752fce7baa2bf6', '5f899a8cbc752fce7ba8cf61', '5f8999e4bc752fce7ba80bff', '5f899ad8bc752fce7ba8fe73', '5f8999ebbc752fce7ba8293f', '5f8999e6bc752fce7ba81762', '5f899aa4bc752fce7ba8dfdc', '5f899c6cbc752fce7ba9ae3e', '5f899dcebc752fce7baa2214', '5f899c73bc752fce7ba9b09d', '5f899ab4bc752fce7ba8e988', '5f899bdebc752fce7ba977d8', '5f899b2fbc752fce7ba92b88', '5f899b0fbc752fce7ba91bc3', '5f899ce6bc752fce7ba9d90f', '5f8999f1bc752fce7ba8375e', '5f899ab1bc752fce7ba8e7f4', '5f899cccbc752fce7ba9d049', '5f899ce6bc752fce7ba9d8f1', '5f899ccabc752fce7ba9cf86', '5f899ceebc752fce7ba9dba5', '5f899cb8bc752fce7ba9c927', '5f899ca5bc752fce7ba9c29f', '5f899db3bc752fce7baa1a01', '5f899b71bc752fce7ba949e7', '5f899adabc752fce7ba8ff87', '5f8999eabc752fce7ba82594', '5f899a8dbc752fce7ba8d05a', '5f899ba4bc752fce7ba95f86', '5f899b76bc752fce7ba94c33', '5f899ba1bc752fce7ba95e0f', '5f899c51bc752fce7ba9a3fa', '5f899b96bc752fce7ba959aa', '5f899d79bc752fce7baa0864', '5f899c26bc752fce7ba99437', '5f899a1abc752fce7ba87129', '5f899af0bc752fce7ba90b82', '5f899b9dbc752fce7ba95c7f', '5f899d59bc752fce7ba9fe77', '5f899a2dbc752fce7ba884b0', '5f899b96bc752fce7ba959a6', '5f899acebc752fce7ba8f896', '5f8999eebc752fce7ba830e7', '5f899b3ebc752fce7ba932e0', '5f899d02bc752fce7ba9e281', '5f899afabc752fce7ba910bb', '5f8999ffbc752fce7ba84fe4', '5f899d08bc752fce7ba9e472', '5f899baabc752fce7ba961e0', '5f8999e4bc752fce7ba80cba', '5f899a09bc752fce7ba85e6c']
def find_origin(ID,list):
    datasetid=[]
    for l in list:
        for k in l.Items:
            if ID==k['ID']:
                datasetid.append(l.id)
    return datasetid
def find_occurence(ID,list):
    occ=[]
    for k in list:
        for it in k.Items:
            if it['ID']==ID:
                occ.append(k.id)
    return occ
allItems=[]
for k in datasets:
    for it in k.Items:
        allItems.append(it['ID'])
c=Counter(allItems)
print(c)
ff=0
double=[]
map={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
for k in c:
    if c[k]>1:
        print(c[k])
        ff+=1
        print(ff)
        double.append(k)
for k in datasets:
    for j in k.Items:
        if j['ID'] in double:
            map[k.id].append(j['ID'])
for ITSS in ITS:
    print(find_origin(ITSS,datasets))
print("occ=",find_occurence(map[0][0],datasets))
print(found)