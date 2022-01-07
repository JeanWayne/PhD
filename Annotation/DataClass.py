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
