#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:31:13 2018

@author: dmitriy
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
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


def func(pct, allvals,nya):
    absolute = int(pct/100.*np.sum(allvals))
    if absolute > nya:
        return "{:.1f}%\n({:d})".format(pct, absolute)
    else:
        return " "

def donut(a,b,title,nya):    
    fig, ax = plt.subplots(figsize=(18, 9), subplot_kw=dict(aspect="equal"))
    
#    dpi = 80
#    fig = plt.figure(dpi = dpi, figsize = (512 / dpi, 384 / dpi) )
#    #fig = plt.figure()
#    mpl.rcParams.update({'font.size': 10})
    wedges, texts, autotexts = ax.pie(a, autopct=lambda pct: func(pct, a, nya),
                                      textprops=dict(color="w"))
    
    ax.legend(wedges, b,
              title="Authors",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.setp(autotexts, size=24, weight="bold")
    
    ax.set_title(title)
    plt.savefig(title + ".png")
    plt.show()

donut(k["topics"],k["authors"],"Topics", 5)
donut(k["comments"],k["authors"],"Comments", 500)
donut(k["avg"],k["authors"],"Avg", 50)

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

def pieplot(values,title):
    
    labels = k["authors"]
        
    trace = go.Pie(labels=labels, values=values)
    
    plotly.offline.init_notebook_mode(connected=True)
    
    plotly.offline.plot({"data": [trace],
        "layout": go.Layout(title=title)
    }, auto_open=False, filename= str(title) + '.html')

pieplot(k["topics"], "Topics")
pieplot(k["comments"], "Comments")
pieplot(k["avg"], "Avg")
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


#layout = go.Layout(
#    xaxis=dict(
#        showgrid=False,
#        showline=False,
#        showticklabels=False,
#        zeroline=False,
#        domain=[0.15, 1]
#    ),
#    yaxis=dict(
#        showgrid=False,
#        showline=False,
#        showticklabels=False,
#        zeroline=False,
#    ),
#    barmode='stack',
#    paper_bgcolor='rgb(248, 248, 255)',
#    plot_bgcolor='rgb(248, 248, 255)',
#    margin=dict(
#        l=120,
#        r=10,
#        t=140,
#        b=80
#    ),
#    showlegend=False,
#)
#
#fig = go.Figure(data=data, layout=layout)
data = [go.Bar(
            x=k["authors"],
            y=k["topics"],
            
)]
layout = dict(title = 'Topics')
fig = go.Figure(data=data, layout=layout)

plotly.offline.plot(fig, filename='horizontal-bar.html', auto_open = False)

data = [go.Bar(
            x=k["authors"],
            y=k["comments"],
            
            
)]
layout = dict(title = 'Comments')
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='horizontal-bar2.html', auto_open = False)

data = [go.Bar(
            x=k["authors"],
            y=k["avg"],
            
            
)]
layout = dict(title = 'Avg')
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='horizontal-bar3.html', auto_open = False)

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
#ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')
#plt.savefig("dick.png")
plt.show()



print(df.head())
df["date"] = df["date"].replace('в ','',regex=True)
df["time"] = df["time"].replace(['Обновлено: ','в '],'',regex=True) 
#df["date"] = pd.to_datetime(df["date"])
#df["time"] = pd.to_datetime(df["time"])



#labels = df["author"]
#
#values = df["date"]
#
#title = "plot"
#        
#trace = go.Pie(labels=labels, values=values)
#
#plotly.offline.init_notebook_mode(connected=True)
#
#plotly.offline.plot({"data": [trace],
#    "layout": go.Layout(title=title)
#}, auto_open=False, filename= str(title) + '.html')    

print(df.head())
print(df.dtypes)
plt.plot(df["date"], df["counts"])
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
#np.random.seed(19680801)

title = "1"

N = len(list)
x = k["topics"]
y = k["authors"]
colors = np.random.rand(N)
area = k["avg"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

title = "2"

N = len(list)
x = k["comments"]
y = k["authors"]
colors = np.random.rand(N)
area = k["topics"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

title = "3"

N = len(list)
x = k["topics"]
y = k["authors"]
colors = np.random.rand(N)
area = k["comments"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

title = "4"

N = len(list)
x = k["avg"]
y = k["authors"]
colors = np.random.rand(N)
area = k["comments"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

title = "5"

N = len(list)
x = k["comments"]
y = k["authors"]
colors = np.random.rand(N)
area = k["avg"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

title = "6"

N = len(list)
x = k["avg"]
y = k["authors"]
colors = np.random.rand(N)
area = k["topics"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

print(df.head(5))
df["date_split"] = df["date"].str.split(" ")
df["ouy"] = df["date_split"].str[1]
df["wow"] = df["date_split"].str[2]
df["time_split"] = df["time"].str.split(" ")
df["ouy2"] = df["time_split"].str[1]
df["wow2"] = df["time_split"].str[2]
wiwi = [*np.unique(df["ouy"])]
mimi = df.groupby(["ouy"]).sum()
#del df["counts"]
df2 = df[["ouy","sometext","wow"]]
nopi = df2.groupby(["ouy", "sometext"]).sum()
nopi = nopi.sort_values(by="wow", ascending=False)
mipi = df2.groupby(["ouy"]).count()
print(nopi)
minnopi = nopi["wow"].min()
maxnopi = nopi["wow"].max()
print("Topics are publish usually from " + minnopi + " to " + maxnopi + " hours")
df3 = df[["ouy","sometext","wow2"]]
nopi2 = df3.groupby(["ouy", "sometext"]).sum()
nopi2 = nopi2.sort_values(by="wow2", ascending=False)
mipi2 = df3.groupby(["ouy"]).count()
print(nopi2)
minnopi2 = nopi2["wow2"].min()
maxnopi2 = nopi2["wow2"].max()
print("Topics are updated usually from " + minnopi2 + " to " + maxnopi2 + " hours")
df4 = df[["ouy","author"]]
nopi3 = df4.groupby(["ouy"]).sum()
mipi3 = df4.groupby(["ouy"]).count()
print(nopi3)
print(wiwi)
print(mimi)
print(mipi)
print(mipi3)
wow3 = round(mimi["counts"]/mipi3["author"], 0)
wow3 = wow3.astype(int)
#wow3 = wow3.sort_values(ascending=False)

plt.plot(wiwi,mimi["counts"])
plt.xticks(rotation=45)
plt.title("Commentsperday")
plt.savefig("commday.png")
plt.show()
plt.plot(wiwi,mipi3["author"])
plt.xticks(rotation=45)
plt.title("Topicsperday")
plt.savefig("topicday.png")
plt.show()
plt.plot(wiwi,wow3)
plt.xticks(rotation=45)
plt.title("DayAvg")
plt.savefig("dayavg.png")
plt.show()

k2 = pd.DataFrame({"days":wiwi, "topics":mipi3["author"], "comments":mimi["counts"], "avg":wow3})
k2.reset_index(drop=True, inplace=True)
avgtop = round(int(k["topics"].sum()/k["topics"].count()))
print("Avg numbers of topics in one day is " + str(avgtop) + " topics")
avgcom = round(int(k["comments"].sum()/k["topics"].count()))
print("Avg numbers of comments in one day is " + str(avgcom) + " comments")
worktime = 12*60*60
freqtop = round(worktime/avgtop)
print("Each new topic appear in " + str(freqtop/60) + "minutes or " + str(freqtop) + " seconds")
freqcom = round(worktime/avgcom)
print("Each new comment appear in " + str(freqcom) + " seconds")
avgfreq = round(freqtop/freqcom)
print("Frequency of appear new comment in " + str(avgfreq) + " times more then frequency of appear new topic")
print(k2.head(5))
print(df.head(5))
nope = df.groupby(["author","ouy"]).count()
nopechunk = nope["counts"]
nope2 = df.groupby(["author","ouy"]).sum()
#print(nope)
#print(nopechunk)
#print(nope2)
bnope = df.groupby(["ouy","author"]).count()
bnopechunk = bnope["counts"]
bnope2 = df.groupby(["ouy","author"]).sum()
#print(bnope)
#print(bnopechunk)
#print(bnope2)
bnipe = pd.DataFrame({"topics":bnope["counts"],"comments":bnope2["counts"]})
print(bnipe)
bnipe2 = pd.DataFrame({"topics":nope["counts"],"comments":nope2["counts"]})
print(bnipe2)
#pi = df.groupby("date").count()
#print(pi)
#pi2 = df.groupby("date").sum()
#print(pi2)
#pi3 = [*np.unique(df["date"])]
#print(pi3)
bnipe.to_csv("summarytable1.csv")
bnipe2.to_csv("summarytable2.csv")
k.to_csv("authorsanalyze.csv")