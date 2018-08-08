#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 18:33:29 2018

@author: dmitriy
"""

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import matplotlib.pyplot as plt
import nltk
df = pd.read_csv("itctray.csv")
del df["Unnamed: 0"]
print(df.head())

df["title"] = df["title"]

df2 = df["title"].tolist()

df3 = df["sometext"].tolist()
#Change df2 to df3 and you will train not on title, but on sometext
list = [*df2]

list2 = []

for i in list:
    
    list2.extend(i.split(" "))

print(list)
print(list2.index("Tesla"))
print(list2)

nltk.download("stopwords")
#import textgenrnn

# Getting the Moby Dick HTML 
r = requests.get('https://itc.ua')

# Setting the correct text encoding of the HTML page
r.encoding = 'utf-8'

# Extracting the HTML from the request object
html = r.text

# Printing the first 2000 characters in html
print(html[0:2000])


# Creating a BeautifulSoup object from the HTML
soup = BeautifulSoup(html, 'html.parser')

# Getting the text out of the soup
text = soup.get_text()

# Printing out text between characters 32000 and 34000
print(text[32000:34000])

print(list)

from nltk.tokenize import RegexpTokenizer

# Creating a tokenizer
tokenizer = RegexpTokenizer('\w+')

# Tokenizing the text
#tokens = tokenizer.tokenize(html)
#tokens = tokenizer.tokenize(list2)

# Printing out the first 8 words / tokens 
#print(tokens[:8])

# A new list to hold the lowercased words
words = list

# Looping through the tokens and make them lower case
#for word in tokens:
#    word = word.lower()
#    words.append(word)
# Printing out the first 8 words / tokens 
#print(words[0:8])

from nltk.corpus import stopwords
# Getting the English stop words from nltk
sw = nltk.corpus.stopwords.words('russian')


# Printing out the first eight stop words
print(sw[0:8])

# A new list to hold Moby Dick with No Stop words
words_ns = []

# Appending to words_ns all words that are in words but not in sw
for word in words:
    if word not in sw:
        words_ns.append(word)

# Printing the first 5 words_ns to check that stop words are gone
print(words_ns[0:5])

#%matplotlib inline

# Creating the word frequency distribution
freqdist = nltk.FreqDist(words_ns)

thelist = words_ns

thefile = open('test.txt', 'w')

for item in thelist:
  thefile.write("%s\n" % item)
# Plotting the word frequency distribution
freqdist.plot(25, cumulative=False)
#freqdist.hapaxes()

thefile.close()

days_file = open('test.txt','r+')

print(days_file.read())
print(days_file.readline())
print(days_file)
days_file = str(days_file)
#from textgenrnn import textgenrnn
#t=textgenrnn('textgenrnn_weights.hdf5')
##textgen = textgenrnn()
#t.generate(5, temperature = 1.0)

#t.train_from_file("test.txt", num_epochs=100)
#textgen.generate(1, temperature = 1.0)

#days_file.close()