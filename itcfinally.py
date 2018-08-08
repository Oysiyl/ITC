#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 18:10:03 2018

@author: dmitriy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:28:26 2018

@author: dmitriy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 16:44:53 2018

@author: dmitriy
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd

data = pd.DataFrame()

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36'}

listadres = []

numbers = range(1,26)
l = [*numbers]
print(l)
for i in numbers:
    if i == 1:
        example = "https://itc.ua/"
    else:
        example = "https://itc.ua/page/" + str(i) + "/"
    listadres.append(example)
    
print(listadres)
def onepage(adres):
    page = requests.get(adres, headers = headers)
    
    if page.status_code != requests.codes.ok:
        return False
    
    soup = BeautifulSoup(page.content, 'html.parser')

    title = soup.find_all("h2", class_ = "entry-title")

    time = soup.find_all("time", class_ = "published")

    timeup = soup.find_all("time", class_ = "screen-reader-text updated")
    
    author = soup.find_all("a", class_ = "screen-reader-text fn")

    counts = soup.find_all("span", class_ = "comments part")

    sometext = soup.find_all("div", class_ = "entry-excerpt hidden-xs")

    listtitle = []    
    listtime = []
    listtimeup = []
    listauthor = []
    listcounts = []
    listsometext = []
    
    for i in range(0,24):
        print(i)
        print(title[i])
        k = title[i].get_text()
        listtitle.append(k)
        l = time[i].get_text()
        listtime.append(l)
        m = timeup[i].get_text()
        listtimeup.append(m)
        n = author[i].get_text()
        listauthor.append(n)
        o = counts[i].get_text()
        listcounts.append(o)
        p = sometext[i].get_text()
        listsometext.append(p)
    
    
    df = pd.DataFrame({
            "title": listtitle,
            "date": listtime,
            "time": listtimeup,
            "author": listauthor,            
            "counts": listcounts,
            "sometext": listsometext},)
        
    return df
    
onepage(listadres[0])

df = pd.DataFrame()

for url in listadres:
    df = pd.concat([df, onepage(url)], ignore_index = True)
    
print(df)
df.to_csv("itctray.csv")
