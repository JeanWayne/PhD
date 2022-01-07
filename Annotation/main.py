from kivy.app import App
from kivy.core.window import Window
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import AsyncImage
from kivy.loader import Loader
from kivy.uix.label import Label
import json
import sys
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.textinput import TextInput

def deThumbnailURL(url):
    url=url.replace("/thumb","")
    cutoff_idx=url.rfind("/")
    url=url[:cutoff_idx]
    return url

class DataSet():
    def __init__(self,json):
        self.Items=[]
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

    def saveJSON(self):
        with open("Label_File_#1.json", mode="w",encoding="utf-8") as file:
            json.dump(self.Items, file, indent=4)
class Item():
    def __init__(self,id,caption,tokens,url,selected):
        self.ID=id
        self.caption=caption
        self.tokens=tokens
        self.URL=url
        self.selected=[]



class ImageButton(ButtonBehavior, AsyncImage):
    pass

def buttonbuilder(txt,call):
    b=aButton(text=txt,
           font_size="20sp",
           color=(1, 1, 1, 1),
           size=(32, 32),
           size_hint=(.2, .3),
            on_press=call)
    return b

def tokenizeCaption(s):
    removelist=[",",".",":",";"]
    for remove in removelist:
        s=s.replace(remove,"")
    split=s.split(" ")
    split=list(set(split))
    return split

class aButton(Button):
    #background_color = [0, 0, 0.2, 1]
    wasPressed=False


class AnnotatorApp(App):
    def on_request_close(self, *args):
        self.ds.saveJSON()
        self.stop() #close app
        return True

    def CaptionButtonPress(self,instance):
        if(instance.wasPressed):
            instance.background_color=[0.2,0.4,0,1]
        else:
            instance.background_color=[0.05,1,0,1]

        print(self.ds.Items[self.INDEX]["selected"])
        if instance.text not in self.ds.Items[self.INDEX]["selected"] and not instance.wasPressed:
            self.ds.Items[self.INDEX]["selected"].append(instance.text)
        elif instance.text  in self.ds.Items[self.INDEX]["selected"] and  instance.wasPressed:
            self.ds.Items[self.INDEX]["selected"].remove(instance.text)
        instance.wasPressed = not instance.wasPressed
        print(self.ds.Items[self.INDEX]["selected"])

    def UpdateGUI(self,idx,instance):
        cap= self.ds.Items[idx]["caption"]
        if len(cap) > 320:
            cap = cap[0:160] + "\n" + cap[160:]
        #in case of error, remove deThumbnailURL
        self.img.source = "http:" + deThumbnailURL(self.ds.Items[self.INDEX]["URL"])
        self.caption_label.text =cap
        self.updateButtons(instance)
        self.ds.saveJSON()

    def back(self,instance):
        if(self.INDEX>0):
            self.INDEX-=1
            self.UpdateGUI(self.INDEX,instance)


    def next(self,instance):
        if(self.INDEX<self.MAX_INDEX):
            self.INDEX+=1
            self.UpdateGUI(self.INDEX,instance)

    def goto(self,instance):
        try:
            val=int(self.idx_txt_input.text)
            if val<self.MAX_INDEX and val >=0:
                self.INDEX=val
                self.UpdateGUI(val,instance)
        except:
            pass


    def updateButtons(self,instance):
        for child in [child for child in self.captionLayout.children]:
            self.captionLayout.remove_widget(child)
        for t in self.ds.Items[self.INDEX]["tokens"]:
            wasSelected=t in self.ds.Items[self.INDEX]["selected"]
            btn = buttonbuilder(t, call=self.CaptionButtonPress)
            btn.wasPressed = wasSelected
            if not wasSelected:
                btn.background_color=[0.2,0.4,0,1]
            else:
                btn.background_color =[0.05,1,0,1]
            self.buttonList.append(btn)
            self.captionLayout.add_widget(btn)
        self.idx_txt_input.text=str(self.INDEX)
        self.updateProgressBar()
        self.searchBar.value=self.INDEX
        if not self.ds.Items[self.INDEX]["unsure"]:
            self.unsure_button.background_color=[0.2,0.4,0,1]
            self.unsure_button.text="Confident"
        else:
            self.unsure_button.background_color=[1.,.05,0.05,1]
            self.unsure_button.text="UNSURE!"

    def toogleStretch(self,instance):
        self.img.allow_stretch=not self.img.allow_stretch

    def unsure(self,instance):
        instance.wasPressed=not instance.wasPressed
        self.ds.Items[self.INDEX]["unsure"]=instance.wasPressed
        self.updateButtons(instance)
        self.ds.saveJSON()

    def countCompletedItems(self):
        count=0
        for s in self.ds.Items:
            if len(s["selected"])>0 or s["unsure"]:
                count+=1
        return count
    def countUncompletedItems(self):
        return self.MAX_INDEX+1-self.countCompletedItems()

    def gotoNext(self,instance):
        next_idx=-1
        for s in range(len(self.ds.Items)):
            if (len(self.ds.Items[s]["selected"])<1):
                next_idx=s
                break
        if next_idx<self.MAX_INDEX and next_idx >=0:
            self.INDEX=next_idx
            self.UpdateGUI(next_idx,instance)

    def updateProgressBar(self):
        self.pb.value=self.countCompletedItems()

    def press(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'left':
            self.back()
        if keycode[1] == 'right':
            self.next()

    def build(self):
        self.INDEX=0
        if len(sys.argv)<2:
            with open('Files/File_0.json') as json_file:
                self.data = json.load(json_file)
        else:
            with open(sys.argv[1]) as json_file:
                self.data = json.load(json_file)

        self.ds = DataSet(self.data)
        self.MAX_INDEX=len(self.ds.Items)-1
        self.buttonList=[]
        caption=self.ds.Items[self.INDEX]["caption"]
        if len(caption)>320:
            caption=caption[0:160]+"\n"+caption[160:]
        #tokens=self.ds.Items[self.INDEX]["tokens"]

        #Create GUI
        self.caption_label = Label(text =caption,size_hint=(1,.10),font_size="14sp")
        self.idx_count_label = Label(text ="of "+str(self.MAX_INDEX),size_hint=(1,.10),font_size="14sp")

        self.pb = ProgressBar(max= self.MAX_INDEX,size_hint=(1,.01))
        self.searchBar = ProgressBar(max= self.MAX_INDEX,size_hint=(1,.01))

        idxLayout=GridLayout(cols=5, size_hint=(1, .05))
        gridlayout = GridLayout(cols=3, size_hint=(1, .90))
        mainLayout = BoxLayout(orientation = 'vertical')
        self.captionLayout = GridLayout(cols=8,size_hint=(1,.20))#BoxLayout(orientation = 'horizontal',size_hint=(1,.20))
        vboxLayout = BoxLayout(orientation = 'vertical',size_hint=(1,.80))

        self.idx_txt_input=TextInput(text=str(self.INDEX), multiline=False)
        self.idx_txt_input.halign="center"

        self.img =AsyncImage(source = "http:"+self.ds.Items[self.INDEX]["URL"])
        self.img.keep_ratio = True
        self.img.allow_stretch=True

        self.toogleStretch_button=Button(text="Stretch",
           font_size="12sp",
           color=(1, 1, 1, 1),
           size=(32, 32),
           size_hint=(1, .05),
            on_press=self.toogleStretch)

        self.goto_button=Button(text="GOTO",
           font_size="10sp",
           color=(1, 1, 1, 1),
           size=(32, 32),
           size_hint=(1, 1),
            on_press=self.goto)

        self.nextNew_button=Button(text=">> NEW  ITEM >>",
           font_size="12sp",
           color=(1, 1, 1, 1),
           size=(32, 32),
           size_hint=(1, 1),
            on_press=self.gotoNext)

        self.button_next=Button(text="N\nE\nX\nT",
           font_size="20sp",
           color=(1, 1, 1, 1),
           size=(32, 32),
           size_hint=(.1, 1.),
            on_press=self.next)
        self.button_back=Button(text="B\nA\nC\nK",
           font_size="20sp",
           color=(1, 1, 1, 1),
           size=(32, 32),
           size_hint=(.1, 1.),
            on_press=self.back)
        self.unsure_button=aButton(text="Confident!",
           font_size="14sp",
           color=(1, 1, 1, 1),
           size=(32, 32),
           size_hint=(1, 1.),
            on_press=self.unsure)
        self.unsure_button.wasPressed=self.ds.Items[self.INDEX]["unsure"]
        #FILL Layout
        idxLayout.add_widget(self.goto_button)
        idxLayout.add_widget(self.idx_txt_input)
        idxLayout.add_widget(self.idx_count_label)
        idxLayout.add_widget(self.nextNew_button)
        idxLayout.add_widget(self.unsure_button)
        #idxLayout.add_widget(self.checkbox)

        gridlayout.add_widget(self.button_back)
        gridlayout.add_widget(self.img)
        gridlayout.add_widget(self.button_next)

        vboxLayout.add_widget(self.pb)
        vboxLayout.add_widget(idxLayout)
        vboxLayout.add_widget(self.searchBar)
        vboxLayout.add_widget(gridlayout)
        vboxLayout.add_widget(self.toogleStretch_button)
        vboxLayout.add_widget(self.caption_label)

        mainLayout.add_widget(vboxLayout)
        mainLayout.add_widget(self.captionLayout)

        self.updateButtons(self)

        Window.bind(on_request_close=self.on_request_close)

        return mainLayout




if __name__ == '__main__':
    Window.size = (1280, 868)
    AnnotatorApp().run()