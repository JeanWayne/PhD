from pymongo import MongoClient
import numpy as np
import math
from matplotlib import pyplot as plt

client=MongoClient('localhost',27017)
db="WikiHarvest"
col_source="harvest_1"
col_target="uniques"
#client[db][col_target].drop()
print("number of wiki pages harvested with images: "+str(client[db][col_source].count_documents({})))
print("Number of Unique Images: "+str(client[db][col_target].count_documents({})))
numOfCaptions=[]
for f in client[db][col_target].find({}):
    numOfCaptions.append(f["result_size"])
numOfCaptions=np.asarray(numOfCaptions)

bins = np.linspace(math.ceil(min(numOfCaptions)),
                   math.floor(max(numOfCaptions)),
                   30) # fixed number of bins

plt.xlim([0, max(numOfCaptions)+1])

plt.hist(numOfCaptions, bins=bins, alpha=0.5)
plt.title('Count for Images with respecting number of captions found')
plt.xlabel('Number of Captions found for one Image')
plt.ylabel('count')

plt.show()