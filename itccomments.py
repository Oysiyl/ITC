#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:31:13 2018

@author: dmitriy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("itctray.csv")
del df["Unnamed: 0"]
print(df.head())
print(df.info())
list = np.unique(df["author"], return_index=False, return_inverse=False, return_counts=False, axis=None)
list = list.tolist()
df.sort_values(["author"], ascending = True)
print(df.head())
print(list)
print(len(list))


name = df.groupby('author')
wow0 = df.groupby('author').sum()
wow = wow0.sort_values(by='counts', ascending=False)
wow2 = df.groupby('author').count()
wow2 = wow2.iloc[:,0:1]
wow2 = wow2.sort_values(by='counts', ascending=False)
wow3 = round(wow["counts"]/wow2["counts"], 0)
wow3 = wow3.astype(int)
wow3 = wow3.sort_values(ascending=False)
k = pd.DataFrame({"authors":list, "topics":wow2["counts"], "comments":wow["counts"], "avg":wow3})
k.reset_index(drop=True, inplace=True)
#k = k.reset_index()
#for i in k:
#    if k["author"]=="Вадим Карпусь":
#        list2.append()

#print(k[0])
print(list[0])
print(name)
print(wow)
print(wow2)
print(wow3)
print(k)
plt.bar(k["authors"],k["topics"])
plt.xticks(rotation=45)
plt.show()


def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    if absolute > 5:
        return "{:.1f}%\n({:d})".format(pct, absolute)
    else:
        return " "

def donut(a,b,title):    
    fig, ax = plt.subplots(figsize=(18, 9), subplot_kw=dict(aspect="equal"))
    
    
    wedges, texts, autotexts = ax.pie(a, autopct=lambda pct: func(pct, a),
                                      textprops=dict(color="w"))
    
    ax.legend(wedges, b,
              title="Authors",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.setp(autotexts, size=24, weight="bold")
    
    ax.set_title(title)
    
    plt.show()

donut(k["topics"],k["authors"],"Topics")
donut(k["comments"],k["authors"],"Comments")
donut(k["avg"],k["authors"],"Avg")

#from bokeh.plotting import *
#from numpy import pi
#
## define starts/ends for wedges from percentages of a circle
#percents = [0, 0.3, 0.4, 0.6, 0.9, 1]
#starts = [p*2*pi for p in percents[:-1]]
#ends = [p*2*pi for p in percents[1:]]
#
## a color for each pie piece
#colors = ["red", "green", "blue", "orange", "yellow"]
#
#p = figure(x_range=(-1,1), y_range=(-1,1))
#
#p.wedge(x=0, y=0, radius=1, start_angle=starts, end_angle=ends, color=colors)
#show(p)

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

def pieplot(values,count):
    
    labels = k["authors"]
        
    trace = go.Pie(labels=labels, values=values)
    
    plotly.offline.init_notebook_mode(connected=True)
    
    plotly.offline.plot({"data": [trace],
        "layout": go.Layout(title="hello world")
    }, auto_open=False, filename='basic_pie_chart' + str(count) + '.html')

pieplot(k["topics"], 1)
pieplot(k["comments"], 2)
pieplot(k["avg"], 3)
#labels = k["authors"]
#values = k["comments"]
#
#trace = go.Pie(labels=labels, values=values)
#
#plotly.offline.init_notebook_mode(connected=True)
#
#plotly.offline.plot([trace], filename='basic_pie_chart1.html')
#labels = k["authors"]
#values = k["avg"]
#
#trace = go.Pie(labels=labels, values=values)
#
#plotly.offline.init_notebook_mode(connected=True)
#
#plotly.offline.plot([trace], filename='basic_pie_chart2.html')
#fig, ax = plt.subplots(figsize=(18, 9), subplot_kw=dict(aspect="equal"))
#
#data = k["comments"]
#
#ingredients = k["authors"]
#
#wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
#                                  textprops=dict(color="w"))
#
#ax.legend(wedges, ingredients,
#          title="Authors",
#          loc="center left",
#          bbox_to_anchor=(1, 0, 0.5, 1))
#
#plt.setp(autotexts, size=24, weight="bold")
#
#ax.set_title("Comments")
#
#plt.show()
#
#
#
#fig, ax = plt.subplots(figsize=(18, 9), subplot_kw=dict(aspect="equal"))
#
#recipe = ["375 g flour",
#          "75 g sugar",
#          "250 g butter",
#          "300 g berries"]
#
#data = [float(x.split()[0]) for x in recipe]
#data = k["avg"]
#ingredients = [x.split()[-1] for x in recipe]
#ingredients = k["authors"]
#
#
#
#wedges, texts, autotexts = ax.pie(data, autopct=lambda pct: func(pct, data),
#                                  textprops=dict(color="w"))
#
#ax.legend(wedges, ingredients,
#          title="Authors",
#          loc="center left",
#          bbox_to_anchor=(1, 0, 0.5, 1))
#
#plt.setp(autotexts, size=24, weight="bold")
#
#ax.set_title("Avgcommentsto1topic")
#
#plt.show()

#fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(aspect="equal"))
#
#recipe = ["225 g flour",
#          "90 g sugar",
#          "1 egg",
#          "60 g butter",
#          "100 ml milk",
#          "1/2 package of yeast"]
#recipe = k["authors"]
#data = [225, 90, 50, 60, 100, 5]
#data = k["topics"]
#
#wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)
#
#bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
#kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
#          bbox=bbox_props, zorder=0, va="center")
#
#for i, p in enumerate(wedges):
#    ang = (p.theta2 - p.theta1)/2. + p.theta1
#    y = np.sin(np.deg2rad(ang))
#    x = np.cos(np.deg2rad(ang))
#    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
#    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
#    kw["arrowprops"].update({"connectionstyle": connectionstyle})
#    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
#                 horizontalalignment=horizontalalignment, **kw)
#
#ax.set_title("Matplotlib bakery: A donut")
#
#plt.show()


plt.bar(k["authors"],k["comments"])
plt.xticks(rotation=45)
plt.show()

#people = k["authors"]
#y_pos = k["avg"]
#fig, ax = plt.subplots()
#plt.bar(people, y_pos, align='center',
#        color='green', ecolor='black')
#ax.set_yticks(y_pos)
#ax.set_yticklabels(people)
#ax.invert_yaxis()
#ax.set_xlabel('How many?')
#ax.set_title('What do you think about it')
##plt.xticks(rotation=45)
#plt.show()

data = [go.Bar(
            x=k["authors"],
            y=k["topics"]
)]

plotly.offline.plot(data, filename='horizontal-bar.html')

data = [go.Bar(
            x=k["authors"],
            y=k["comments"]
            
)]

plotly.offline.plot(data, filename='horizontal-bar2.html')

data = [go.Bar(
            x=k["authors"],
            y=k["avg"]
            
)]

plotly.offline.plot(data, filename='horizontal-bar3.html')

fig, ax = plt.subplots()

# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

ax.barh(y_pos, performance, xerr=error, align='center',
        color='green', ecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

plt.show()

    
save = k.to_csv("authorsanalyze.csv")