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

numbers = range(2,12)
l = [*numbers]
print(l)
for i in numbers:
    example = "https://itc.ua/page/" + str(i) + "/"
    listadres.append(example)

print(listadres)
def onepage(adres):
    page = requests.get(adres, headers = headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    possible_links = soup.find_all('a')
    for link in possible_links:
        if link.has_attr('href'):
            print (link.attrs['href'])
            list.append(link.attrs['href'])
    #possible_links = list(possible_links)
    text = possible_links
    print(text)
    return text
    
onepage(listadres[0])


'''
text = range(0,5)
for txt in range(0,5):
    adres = "https://itc.ua/page/" + str(txt) + "/"
    page = requests.get(adres, headers = headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    possible_links = soup.find_all('a')
    for link in possible_links:
        if link.has_attr('href'):
            print (link.attrs['href'])
            list.append(link.attrs['href'])
            list4 = [16,22,28,34,40,46,52,58,64,70,76,82,88,94,100,106,112,118,124,130,136,143,154]
            print(list4)
            print(list)
            print(len(list))
'''            
                
            
"""
print(list)
adres2 = list[16]
page = requests.get(adres2, headers = headers)
soup = BeautifulSoup(page.content, 'html.parser')
title = soup.find("div", class_ = "h1")
up = soup.find("div", class_ = "post-txt")
up = up.get_text()
print(up)
title = title.get_text()
print(title)
comment = soup.find("span", class_ = "comments part")
comment = comment.get_text()
print(comment)
author = soup.find("span", class_ = "vcard author part hidden-xs")
author = author.get_text()
print(author)
date = soup.find("span", class_ = "date part")
date = date.get_text()
print(date)
list2 = []
img = soup.find_all("img")
for link in img:
    if link.has_attr('src'):
        print (link.attrs['src'])
        list2.append(link.attrs['src'])
#img = img.get_text()
print(img)
print(list2[1])
related = soup.find("div", class_ = "itc-related-posts")
related = related.get_text()
print(related)
print(list)
print(len(list))
list4 = [16,22,28,34,40,46,52,58,64,70,76,82,88,94,100,106,112,118,124,130,136,143,154]
print(list4)
list5 = []
for i in list4:
    list5.append(list[i])
print(list5)
"""
'''
24 items
16 first + 6 each
22
28
34
40
'''

#up = list(possible_links)
#print(up[0])
        
'''
up = soup.find_all("a", class_ = "entry-title")
up = list(up)
#up =up.get_text()
print(up[0])
'''
"""
up = soup.find("div", class_ = "thumb")
title = soup.find("span", class_ = "title")
title = title.get_text()
date = soup.find("span", class_ = "date")
date = date.get_text()
duration = soup.find("span", class_ = "duration")
duration = duration.get_text()
view = soup.find("span", class_ = "view")
view = view.get_text()
link = soup.find("span", onclick_ = "")
print(data)
ahref = soup.find('a')
ahref.get('href')
print(ahref)
link = link
link = list(link)
"""
'''
filename = image["src"].split("/")[-1].split("?")[0].replace("$",'').replace(".JPG",".jpg").replace("~~_26",str(count)).lstrip("(")
'''
#link = link.get_text()
#link = link.get_text()
"""
img = soup.img['src']

#img = img.get_text()

print(up)
print(title)
print(date)
print(duration)
print(view)

print(link)

print(img)
#print(img)
up = list(up)
up = up[0]
df = pd.DataFrame({"wow":up}, index = False)
#data.append(soup)
print(up)
print(data)
print(df)

month = [31,28,31,30,31,30,31,31,30,31,30,31]

#month2 = [sum(month[:i+1]) for i in range(len(month))]

#print(month2)

#month2 = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

city = "харьков"
y = 2017
m = 1
d = 1

def prepareadres(city,y,m,d):
    months = [31,28,31,30,31,30,31,31,30,31,30,31]
    d = d + 1
    d0 = 0
    m0 = 0
    for month in months:
        if d > month:
             d = d - month
             m += 1
    if d == sum(months):    
        print("year complete")
        #print(df.iloc[0, :-2])
    if (m < 10) and (d < 10):
        adres = "https://sinoptik.ua/погода-" + city + "/" + str(y) + "-" + str(m0) + str(m) + "-" + str(d0) + str(d)
    elif (m < 10) and (d >= 10):
        adres = "https://sinoptik.ua/погода-" + city + "/" + str(y) + "-"+ str(m0) + str(m) + "-" + str(d)
    elif (m >= 10) and (d < 10):
        adres = "https://sinoptik.ua/погода-" + city + "/" + str(y) + "-" + str(m) + "-" + str(d0) + str(d)   
    else:
        adres = "https://sinoptik.ua/погода-" + city + "/" + str(y) + "-" + str(m) + "-" + str(d)
    #print(adres)
    return adres

year = list(range(0,365))

adres2 = [prepareadres(city,y,m,i) for i in range(len(year))]

print(adres2)
print(len(adres2))

def weather(adres):
    
    page = requests.get(adres)
    soup = BeautifulSoup(page.content, 'html.parser')
    up = soup.find("p", class_ = "infoHistoryval")
    up = list(up)
    infoDayweek = soup.find("p", class_ = "infoDayweek")
    infoDayweek = list(infoDayweek)[0]

    infoDate = soup.find("p", class_ = "infoDate")
    infoDate = list(infoDate)[0]
    infoMonth = soup.find("p", class_ = "infoMonth")
    infoMonth = list(infoMonth)[0]
    infoDaylight = soup.find(class_ = "infoDaylight")
    infoDaylight = list(infoDaylight)

    sunUp = infoDaylight[1].get_text()
    sunDown = infoDaylight[3].get_text()

    maxTemp = up[3].get_text()
    maxYear = up[4]
    maxYear = maxYear.replace("(", "")
    maxYear = maxYear.replace(")", "")

    minTemp = up[9].get_text()
    minYear = up[10]
    minYear = minYear.replace("(", "")
    minYear = minYear.replace(")", "")

    h2 = soup.find_all("div", class_ = "description")
    dayInfo = list(h2)[0].get_text()
    dayHistory = list(h2)[1].get_text()

    tempMax = soup.find("div", class_ = "max")
    tempMax = list(tempMax)[1].get_text()
    tempMin = soup.find("div", class_ = "min")
    tempMin = list(tempMin)[1].get_text()  
    
    df1 = pd.DataFrame({
    "infoDayweek": infoDayweek,
    "infoDate": infoDate,
    "infoMonth": infoMonth,
    "sunUp": sunUp,
    "sunDown": sunDown,
    "tempMax": tempMax,
    "tempMin": tempMin,
    "maxTemp": maxTemp, 
    "maxYear": maxYear, 
    "minTemp": minTemp, 
    "minYear": minYear,
    "dayInfo": dayInfo,
    "dayHistory": dayHistory
    }, index=[0])
    check = df1.iloc[0, 0:-2]
    print(check)

    return df1
    


df = pd.DataFrame()

for url in adres2:
    df = pd.concat([df, weather(url)], ignore_index = True)
    
print(df)
print(df.iloc[:, 0:-2])
df.to_csv("results3.csv")
df = pd.read_csv("results3.csv")
#df2 = df.iloc[:,1:-2]
print(df)
#df2.to_csv("results.csv")
"""    
    

