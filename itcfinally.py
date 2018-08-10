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
#import all the necessary libraries
import pandas as pd
from bs4 import BeautifulSoup
import requests
#Create list which contains how many pages you want to scraping
numbers = [*range(1, 26)]
#Create an empty list for all of the page adresses
listadres = []
#Use for loop to fill list above with page adresses
for i in numbers:
    if i == 1:
        example = "https://itc.ua/"
    else:
        example = "https://itc.ua/page/" + str(i) + "/"
    listadres.append(example)
#Check output    
print(listadres)
#Create function that scraping full one page from site
def onepage(adres):
    #Use headers to prevent hide our script
    headers = {'User-Agent': 'Mozilla/5.0'}
    #Get page
    page = requests.get(adres, headers = headers)
    #Get all of the html code 
    soup = BeautifulSoup(page.content, 'html.parser')
    #Find title the topic
    title = soup.find_all("h2", class_ = "entry-title")
    #Find time when topic is published 
    time = soup.find_all("time", class_ = "published")
    #Find time when topic is updated
    timeup = soup.find_all("time", class_ = "screen-reader-text updated")
    #Find author the topic
    author = soup.find_all("a", class_ = "screen-reader-text fn")
    #Find how many comments have topic
    counts = soup.find_all("span", class_ = "comments part")
    #Find preface for topic
    sometext = soup.find_all("div", class_ = "entry-excerpt hidden-xs")
    #Create an empty lists
    listtitle = []    
    listtime = []
    listtimeup = []
    listauthor = []
    listcounts = []
    listsometext = []
    #Fill the lists above our scraping date
    for i in range(0, 24):
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
    #Create DataFrame, that will contains info from lists
    df = pd.DataFrame({
            "title": listtitle,
            "date": listtime,
            "time": listtimeup,
            "author": listauthor,            
            "counts": listcounts,
            "sometext": listsometext})
    #Function will return that DataFrame    
    return df
    
#Create an empty DataFrame
df = pd.DataFrame()
#Adding each new page in one DataFrame
for url in listadres:
    df = pd.concat([df, onepage(url)], ignore_index = True)
#Check df
print(df)
#Save DataFrame to csv
df.to_csv("itctray.csv")