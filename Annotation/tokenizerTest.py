
import re
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

from transformers import RobertaTokenizer

from Annotation.EvaluationFuncs import evaluate3

a="LâĢĵ US 221 Truck/US 321 Truck/US 421 Truck/NC 105 traveling concurrent through the Boone area"
print(a)
def replaceUnicodeError(s):
    liste=[("Äį","č"),("Äĩ","ć"),("ÃŃ","í"),("Ã©","é"),("Ãī","É"),("âĢĵ","–")]
    for l in liste:
        s=s.replace(l[0],l[1])
    return s
b=replaceUnicodeError(a)
print(b)
print()


str1="Dr. Valentine Mott by Anson Dickinson c. 1820"
allTokens1= ['Dr.', 'Valentine', 'Mott', 'Anson', 'Dickinson', 'c.', '1820']
rob1: [('Dickinson', 0.6240161089115729), ('Mott', 0.5780774866846817), ('Valentine', 0.5611065000549115), ('Anson', 0.46316457280792234), ('1820', 0.4413064949893354), ('by', 0.4160448405129412), ('c', 0.4135059159147383)]

str2="Logo of the SC Department of Natural Resources"
allTokens2= [    "Logo",    "SC",    "Department",    "Natural",    "Resources"]
rob2= [('Resources', 0.488388851789447), ('Natural', 0.4744665902066376), ('SC', 0.45316774305787727), ('Department', 0.42078712145961455), ('of', 0.34066676377828803), ('the', 0.3036776613333218), ('of', 0.27173250315391195)]


str3="The rebellion of Gy\u00f6rgy D\u00f3zsa in 1514 spread like lightning in the Kingdom of Hungary where hundreds of manor-houses and castles were burnt and thousands of the gentry killed by impalement, crucifixion and other methods. D\u00f3zsa is here depicted punished with heated iron chair and crown"
allTokens3= ['The', 'rebellion', 'György', 'Dózsa', '1514', 'spread', 'like', 'lightning', 'Kingdom', 'Hungary', 'hundreds', 'manor-houses', 'castles', 'burnt', 'thousands', 'gentry', 'killed', 'impalement', 'crucifixion', 'methods', 'Dózsa', 'depicted', 'punished', 'heated', 'iron', 'chair', 'crown']
rob3= [('lightning', 0.33812881318763593), ('GyÃ¶rgy', 0.32329980087142723), ('DÃ³zsa', 0.3046428395323292), ('Hungary', 0.27975836528431886), ('manor-houses', 0.2774053292370019), ('rebellion', 0.2554593636291212), ('Kingdom', 0.252655347977582), ('like', 0.25007013524762967), ('chair', 0.24518322693585937), ('1514', 0.2448683086386154), ('castles', 0.2425672153694904), ('DÃ³zsa', 0.2338300404276582), ('spread', 0.23147304156806237), ('hundreds', 0.2263158465714133), ('iron', 0.22116551883248442), ('heated', 0.21719695006383208), ('and', 0.2091182732197851), ('methods', 0.20827108512338827), ('where', 0.20715174719294904), ('burnt', 0.20423753409081), ('crown', 0.20395565958183612), ('in', 0.19611034331388688), ('here', 0.19605201749419351), ('depicted', 0.19573002782860957), ('of', 0.19430025060656983), ('punished', 0.1862382305442311), ('and', 0.1811846939544055), ('gentry', 0.17959890757177505), ('in', 0.1794807393162046), ('other', 0.17902758008520842), ('of', 0.17342142526675686), ('thousands', 0.17214896967778348), ('the', 0.1666690813505002), ('were', 0.1642110338552378), ('impalement', 0.16171828813408423), ('killed', 0.16113403307307247), ('of', 0.16035172707242532), ('and', 0.1567795804374847), ('with', 0.15438692879414312), ('and', 0.15251787157207478), ('crucifixion', 0.15080909008271368), ('is', 0.14561652964524308), ('of', 0.13683798670831623), ('by', 0.1330842681876068), ('the', 0.1300724283604116)]


str4="Un aurin \u00e1quil. Un person con un aquilin nase have un nase quel es simil."
allTokens4= ['Un', 'aurin', 'áquil', 'Un', 'person', 'con', 'un', 'aquilin', 'nase', 'un', 'nase', 'quel', 'es', 'simil']
rob4= [('aurin', 0.27837847518821945), ('Ã¡quil', 0.26307669179109705), ('simil', 0.25127095009430556), ('aquilin', 0.23888875588154265), ('quel', 0.2347290045369893), ('nase', 0.2275487674313743), ('person', 0.22714453194029846), ('un', 0.22608682486177373), ('Un', 0.22581918619661806), ('es', 0.22358131279998986), ('con', 0.22232897117672623), ('nase', 0.21975586209113585), ('un', 0.21881910507813918), ('have', 0.1853554730554432)]

str5= "The pandemic shutdown has shown us the problem. It has revealed what the world looks like without as much pollution, without the chaos and roar of mostly meaningless \"work\" performed by the exploited, using materials stolen from the abused, for the benefit of the pampered and oblivious. Another world is possible, and we've just gotten a glimpse of it. ~Lee Camp"
allTokens5= ["The",    "pandemic",    "shutdown",    "shown",    "us",    "problem",    "It",    "revealed",    "world",    "looks",    "like",    "without",    "much",    "pollution",    "without",    "chaos",    "roar",    "mostly",    "meaningless",    "work",    "performed",    "exploited",    "using",    "materials",    "stolen",    "abused",    "benefit",    "pampered",    "oblivious",    "Another",    "world",    "possible",
    "'ve",    "gotten",    "glimpse",    "~Lee",    "Camp"],
rob_tokens=['<s>', 'ĠThe', 'Ġpand', 'emic', 'Ġshutdown', 'Ġhas', 'Ġshown', 'Ġus', 'Ġthe', 'Ġproblem', '.', 'ĠIt', 'Ġhas', 'Ġrevealed', 'Ġwhat', 'Ġthe', 'Ġworld', 'Ġlooks', 'Ġlike', 'Ġwithout', 'Ġas', 'Ġmuch', 'Ġpollution', ',', 'Ġwithout', 'Ġthe', 'Ġchaos', 'Ġand', 'Ġroar', 'Ġof', 'Ġmostly', 'Ġmeaningless', 'Ġ"', 'work', '"', 'Ġperformed', 'Ġby', 'Ġthe', 'Ġexploited', ',', 'Ġusing', 'Ġmaterials', 'Ġstolen', 'Ġfrom', 'Ġthe', 'Ġabused', ',', 'Ġfor', 'Ġthe', 'Ġbenefit', 'Ġof', 'Ġthe', 'Ġp', 'am', 'pered', 'Ġand', 'Ġoblivious', '.', 'ĠAnother', 'Ġworld', 'Ġis', 'Ġpossible', ',', 'Ġand', 'Ġwe', "'ve", 'Ġjust', 'Ġgotten', 'Ġa', 'Ġglimpse', 'Ġof', 'Ġit', '.', 'Ġ~', 'Lee', 'ĠCamp', '</s>']
cap=[str1,str2,str3,str4,str5]
all=[allTokens1,allTokens2,allTokens3,allTokens4,allTokens5]
stoplist = set(stopwords.words('english') + list(punctuation)+ ["``","''","'s"])

def removeDuplicates(lst):
    return [t for t in (set(tuple(i) for i in lst))]


def annotator_tokenizer(str):
    split = word_tokenize(str)
    split = [s for s in split if s not in stoplist]
    split = [s for s in split if len(s) > 1]
    return split

def RobertFilterTokens(rob,all):
    rob=RobertaFilterString(rob)
    rob=removeDuplicates(rob)
    rob=[(r[0],r[1]) for r in rob if len(r[0]) >1]
    rob=[(r[0],r[1]) for r in rob if r[0].lower() not in stoplist]
    rob=[(r[0].encode("latin-1").decode("utf-8"),r[1]) for r in rob if r[0] not in all]
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

def RemoveRobertaTokens(tokens):
    #tokens.sort(key=lambda tup: tup[1], reverse=True)
    1+1
def fuseTokens(tokens):
    newList=["<s>"]
    if tokens[1][0]!='Ġ':
        tokens[1]='Ġ'+tokens[1]
    IndexList=[0]
    wordCount=0
    for i in range(1,len(tokens)):
        if tokens[i] in ["</s>"] or tokens[i][0]=="Ġ":
            wordCount+=1
        IndexList.append(wordCount)
    numOfWords=IndexList[-2:][0]
    for i in range(1,numOfWords+1):
        try:
            first_pos = IndexList.index(i)
            last_pos = len(IndexList) - IndexList[::-1].index(i) - 1
        except ValueError:
            1+1
        #print(tokens[first_pos], tokens[last_pos])
        if first_pos!=last_pos:
            newList.append(''.join(tokens[first_pos:last_pos+1]))
        else:
            newList.append(tokens[first_pos])
    newList.append("</s>")
    return newList,IndexList

from Experimental.RankWords import getRankingFromRoberta




rob_ranking = getRankingFromRoberta(cap,sort_result=False,model_path="../PyTorch/Model/Epoch_1_2022-01-06_18-52-20_AUC_0.8593628700013282") #AUC :  0.843
tokenizer = RobertaTokenizer.from_pretrained("../PyTorch/Model/Epoch_1_2022-01-06_18-52-20_AUC_0.8593628700013282")
#tokenizer = RobertaTokenizer.from_pretrained("../PyTorch/Model/Epoch_2_2022-01-06_20-39-28_AUC_0.8567258467979885")

for val in range(0,len(cap)):
    s=cap[val]
    inputs = tokenizer.encode_plus(s, return_tensors='pt', add_special_tokens=True)
    input_ids = inputs['input_ids']
    input_id_list = input_ids[0].tolist()  # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    #print(input_ids)
    print(tokens)
    print(RobertFilterTokens(rob_ranking[val],all[val][0]))
    li,idx=fuseTokens(tokens)
    print(len(li),"  ",li)
    print(len(idx),"  ",idx)
    print(s)
    li=li[1:-1]
    li=[i[1:] for i in li]
    fused_tokens=' '.join(li)
    print(fused_tokens)
    print(all[val][0])
    print(annotator_tokenizer(fused_tokens))
    print("____")
    #print(evaluate3(all[val],)
#rob_ranking=[RobertaFilterString(k) for k in rob_ranking]

