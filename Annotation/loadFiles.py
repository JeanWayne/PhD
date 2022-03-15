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

ITS=[]
with open("unique_IDS.txt") as file:
    for lines in file.readlines():
        ITS.append(lines.replace("\n",""))

#ITS=['5f899abfbc752fce7ba8f055', '5f899c55bc752fce7ba9a5aa', '5f899bddbc752fce7ba97711', '5f899c7bbc752fce7ba9b3d5', '5f899a2ebc752fce7ba8854e', '5f899ad1bc752fce7ba8fa5b', '5f899cd7bc752fce7ba9d404', '5f8999f2bc752fce7ba83a65', '5f899b07bc752fce7ba917dc', '5f899cf2bc752fce7ba9dd2c', '5f899a12bc752fce7ba86a0e', '5f899a1ebc752fce7ba87613', '5f899c73bc752fce7ba9b097', '5f899a4ebc752fce7ba8a203', '5f8999fdbc752fce7ba84d79', '5f899c55bc752fce7ba9a580', '5f899a44bc752fce7ba89937', '5f899a02bc752fce7ba85572', '5f899c6fbc752fce7ba9af47', '5f899ad3bc752fce7ba8fbc8', '5f899b6cbc752fce7ba94779', '5f899a0abc752fce7ba85fb6', '5f899ad0bc752fce7ba8f9b8', '5f899cacbc752fce7ba9c4fc', '5f899af5bc752fce7ba90e0c', '5f899a61bc752fce7ba8b0b3', '5f899caebc752fce7ba9c5c3', '5f899d18bc752fce7ba9e9ab', '5f899a24bc752fce7ba87be2', '5f899a48bc752fce7ba89cf9', '5f899b0ebc752fce7ba91b4b', '5f899debbc752fce7baa2ae3', '5f899a22bc752fce7ba879eb', '5f899c36bc752fce7ba99a46', '5f899acfbc752fce7ba8f985', '5f899cf5bc752fce7ba9de31', '5f899b25bc752fce7ba926d8', '5f899c76bc752fce7ba9b1f3', '5f899cc9bc752fce7ba9cef0', '5f899a42bc752fce7ba897df', '5f899a81bc752fce7ba8c7fd', '5f899dafbc752fce7baa18b6', '5f899d07bc752fce7ba9e3f7', '5f899cc6bc752fce7ba9ce18', '5f899ad4bc752fce7ba8fc45', '5f899d29bc752fce7ba9ef36', '5f899ceebc752fce7ba9dbb5', '5f899d18bc752fce7ba9e9be', '5f899a22bc752fce7ba87a51', '5f899b07bc752fce7ba917c1', '5f899ccdbc752fce7ba9d066', '5f899a19bc752fce7ba8702e', '5f899a35bc752fce7ba88c2c', '5f899cb9bc752fce7ba9c993', '5f899c46bc752fce7ba9a006', '5f899cd2bc752fce7ba9d249', '5f899c7bbc752fce7ba9b3b5', '5f899af3bc752fce7ba90d49', '5f899a46bc752fce7ba89b80', '5f899caabc752fce7ba9c464', '5f899addbc752fce7ba90133', '5f899d2dbc752fce7ba9f05c', '5f899af6bc752fce7ba90e9e', '5f899c2dbc752fce7ba996b9', '5f899a3abc752fce7ba8904e', '5f899c9bbc752fce7ba9bf3b', '5f899b20bc752fce7ba92429', '5f899d0fbc752fce7ba9e6ad', '5f899c4abc752fce7ba9a148', '5f899a2abc752fce7ba88220', '5f899a47bc752fce7ba89c01', '5f899c83bc752fce7ba9b6b8', '5f899c7cbc752fce7ba9b3fa', '5f899d09bc752fce7ba9e4b4', '5f899a44bc752fce7ba899d3', '5f899d0fbc752fce7ba9e6a9', '5f899b33bc752fce7ba92d5d', '5f899a35bc752fce7ba88c18', '5f899cb9bc752fce7ba9c95c', '5f899a0abc752fce7ba85f56', '5f899c77bc752fce7ba9b25f', '5f899cf6bc752fce7ba9de52', '5f8999fbbc752fce7ba84a3d', '5f899cddbc752fce7ba9d615', '5f899b2bbc752fce7ba92965', '5f899c50bc752fce7ba9a3eb', '5f899bf8bc752fce7ba98226', '5f899acfbc752fce7ba8f962', '5f899a93bc752fce7ba8d4f5', '5f899cadbc752fce7ba9c542', '5f899df7bc752fce7baa2e6f', '5f899cabbc752fce7ba9c4ba', '5f899c96bc752fce7ba9bd46', '5f899c86bc752fce7ba9b7ae', '5f899a48bc752fce7ba89c94', '5f899b06bc752fce7ba9171b', '5f899ca5bc752fce7ba9c29d', '5f899b16bc752fce7ba91f19', '5f899a0abc752fce7ba85f4e', '5f899dd7bc752fce7baa24f7', '5f899a12bc752fce7ba86a25', '5f899cc8bc752fce7ba9ced9', '5f899acfbc752fce7ba8f9a6', '5f899a6fbc752fce7ba8bb41', '5f899d0cbc752fce7ba9e5aa', '5f899a35bc752fce7ba88c39', '5f8999e8bc752fce7ba81f3d', '5f899aefbc752fce7ba90b15', '5f899ccfbc752fce7ba9d125', '5f899c5bbc752fce7ba9a806', '5f899a6abc752fce7ba8b751', '5f899c46bc752fce7ba99fee']
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
        #print(c[k])
        ff+=1
        #print(ff)
        double.append(k)
for k in datasets:
    for j in k.Items:
        if j['ID'] in double:
            map[k.id].append(j['ID'])
for ITSS in ITS:
    print(find_origin(ITSS,datasets))
#print("occ=",find_occurence(map[0][0],datasets))
countit=0
for gg in c:
    if c[gg]>1:
        countit+=1
print(countit)
print(found)