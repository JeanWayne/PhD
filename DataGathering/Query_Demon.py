import requests
import os,time
import datetime
import random
import json
from bs4 import BeautifulSoup
from urllib.request import urlopen
from pymongo import MongoClient

import re
#import wikipedia

client=MongoClient('localhost',27017)
db="WikiHarvest"
randomURL="https://en.wikipedia.org/wiki/Special:Random"
action='query'
format='json'
prop='globalusage'
gusite='enwiki|enwikibooks|enwikinews|enwikiquote|enwikisource|enwikiversity|enwikivoyage|enwiktionary'
guprop='pageid|url'
gulimit='500'
generator='random'
grnnamespace='6'
last = datetime.datetime.now()
print(last.strftime('%Y-%m-%d  %H:%M %S'))
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
def getCaptionOfUrl(url,imgName):
   ret = requests.get(url, headers=headers)
   soup = BeautifulSoup(ret.text, 'html.parser')
   #print(url)
   images = []
   gally= soup.findAll('li',"gallerybox")
   for g in gally:
      ret = g.find_all("a", {"href":re.compile(imgName)})
      for r in ret:
         image_and_caption = {
            'image_url': url,
            'image_caption': g.text.strip('\n')
         }
         images.append(image_and_caption)
   if len(images)<1:
      thumb_divs = soup.findAll("div", {"class": "thumbinner"})
      for div in thumb_divs:
         try:
            if imgName in div.img.get('src'):
               image_and_caption = {
                  'image_url': url,
                  'image_caption': div.text.strip('\n')
               }
               images.append(image_and_caption)
         except AttributeError:
            print("Error at:" + imgName + "     ,   " + url + "  Error Type: div thumbinner")
            image_and_caption = {"'image_url'": url,
                            "ERROR": "AttributeError @ div thumbinner"
                            }
   if len(images)<1:
      infobox = soup.findAll("a",{"href":re.compile(imgName),"class":"image"})
      for i in infobox:
         if imgName in i.img['src']:
            try:
               image_and_caption = {
                  'image_url': url,
                  'image_caption': i.next.next.text.strip('\n')
               }
               images.append(image_and_caption)
            except AttributeError:
               print("Error at:"+imgName+"     ,   "+url+"  Error Type: mainInfo")
               image_and_caption={"'image_url'":url,
               "ERROR":"AttributeError @ mainInfo -> NO CAPTION!"
               }
   #print(url)
   return images
#def getCaptionOfUrl(url,imgName):
def getWikiCommmonsPage(imgURL):
   imgName=imgURL.split('/')[-2]
   wc_result={}
   wiki_results=[]
   wc_result["wc_Name"]=imgName
   wc_result['wc_URL']=imgURL
   url = 'https://commons.wikimedia.org/w/api.php?action=' + action + '&titles=File:' + imgName + '&format=' + format + '&prop=' + prop + '&gusite=' + gusite + '&guprop=' + guprop + '&gulimit=' + gulimit
   res = requests.get(url, headers=headers)
   # print(res.text)
   try:
      j = json.loads(res.text)
   except json.decoder.JSONDecodeError:
      print("JSON Decoder Error: "+res.text)
      return {"Status":"Json Decoder Error"}
   #print(j)
   try:
      for p in j['query']['pages']:
         c=(j['query']['pages'][p]['globalusage'])
         if len(c)>1 and len(c)<100: #skip images with only one occurence and more than 100
            for ele in c:
               #print(imgURL)
               if "Portal:" not in ele['url'] and "User:" not in ele['url']:
                  res=getCaptionOfUrl(ele['url'],imgName)
                  if res != []:
                     wiki_results.append(res)
      wc_result["result_size"] = len(wiki_results)
      wc_result["result"]=wiki_results
   except KeyError:
      print("Key Error @"+str(imgName)+"  "+imgURL)
   return wc_result
# wikipage = wikipedia.page("Glacier")
# print("Page Title: %s" % wikipage.title)
# print("Page URL: %s" % wikipage.url)
# print("Nr. of images on page: %d" % len(wikipage.images))
# print(" - Main Image: %s" % wikipage.images[0])
# print("")
# print(wikipage.images)
workload=[]
for i in range(1000000):
   #html = urlopen("https://en.wikipedia.org/wiki/Glacier")
   html=urlopen(randomURL)
   bs = BeautifulSoup(html, 'html.parser')
   images = bs.find_all('img', {'src':re.compile('.JPG|.jpg')})
   item=html.url
   result=[]
   last = datetime.datetime.now()
   print(item+"   "+last.strftime('%Y-%m-%d  %H:%M %S'))
   for image in images:
      #print(image)
      if "commons" in image['src']:
         #print(image['src']+'\n')
         #if "Glacier_as_seen_by_ctx" in image['src']:
         #if "Baltoro_glacier_from_air" in image['src']:
         res=getWikiCommmonsPage(image['src'])
         try:
            if res['result_size']>1:
               result.append(res)
         except KeyError:
            print(res)
   consumed = datetime.datetime.now()-last
   if len(result)>0:
      item={"wikiItem":item,"imgCount":len(result),"images":result}
      client[db]["harvest_1"].insert_one(item)
      workload.append(item)
      print("Done.  Found: " + str(len(result)) + ".      --- Seconds needed: " + str(consumed.total_seconds()))

with open('wiki.json', 'w') as json_file:
   json.dump(workload, json_file)
print(workload)