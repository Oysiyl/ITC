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

import datetime

import plotly
import plotly.plotly as py
import plotly.graph_objs as go

df0 = pd.read_csv("itctray.csv")
df = df0
del df["Unnamed: 0"]
print(df.head())
print(df.info())
list = np.unique(df["author"], return_index=False, return_inverse=False, return_counts=False, axis=None)
list = list.tolist()
df.sort_values(["author"], ascending = True)
print(df.head())
print(list)
print(len(list))

w = (32,24)
z = 80

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

donut(k["topics"],k["authors"],"Topics", 16)
donut(k["comments"],k["authors"],"Comments", 1000)
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
            text=k["topics"],
               textposition = 'auto',
               hovertext = "text",
               marker=dict(
                   color='rgb(158,202,225)',
                   line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                    ),
                opacity=0.6
)]
layout = go.Layout(title='Topics',
                   xaxis=dict(title='Author'),
                   yaxis=dict(title='Counts'))
fig = go.Figure(data=data, layout=layout)

plotly.offline.plot(fig, filename='horizontal-bar.html', auto_open = False)

data = [go.Bar(
            x=k["authors"],
            y=k["comments"],
            text=k["comments"],
               textposition = 'auto',
               hovertext = "text",
               marker=dict(
                   color='rgb(158,202,225)',
                   line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                    ),
                opacity=0.6
            
)]
layout = go.Layout(title='Comments',
                   xaxis=dict(title='Author'),
                   yaxis=dict(title='Counts'))
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='horizontal-bar2.html', auto_open = False)

data = [go.Bar(
            x=k["authors"],
            y=k["avg"],
            text=k["avg"],
               textposition = 'auto',
               hovertext = "text",
               marker=dict(
                   color='rgb(158,202,225)',
                   line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                    ),
                opacity=0.6
            
)]
layout = dict(title = 'Avg',
              xaxis=dict(title='Author'),
              yaxis=dict(title='Counts'))
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
plt.figure(figsize=w, dpi=z)
N = len(list)
x = k["topics"]
y = k["authors"]
colors = np.random.rand(N)
area = k["avg"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

title = "2"
plt.figure(figsize=w, dpi=z)
N = len(list)
x = k["comments"]
y = k["authors"]
colors = np.random.rand(N)
area = k["topics"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

title = "3"
plt.figure(figsize=w, dpi=z)
N = len(list)
x = k["topics"]
y = k["authors"]
colors = np.random.rand(N)
area = k["comments"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

title = "4"
plt.figure(figsize=w, dpi=z)
N = len(list)
x = k["avg"]
y = k["authors"]
colors = np.random.rand(N)
area = k["comments"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

title = "5"
plt.figure(figsize=w, dpi=z)
N = len(list)
x = k["comments"]
y = k["authors"]
colors = np.random.rand(N)
area = k["avg"]  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig(title + ".png")
plt.show()

title = "6"
plt.figure(figsize=w, dpi=z)
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
plt.figure(figsize=w, dpi=z)
plt.plot(wiwi,mimi["counts"])
plt.xticks(rotation=45)
plt.title("Commentsperday")
plt.savefig("commday.png")
plt.show()
plt.figure(figsize=w, dpi=z)
plt.plot(wiwi,mipi3["author"])
plt.xticks(rotation=45)
plt.title("Topicsperday")
plt.savefig("topicday.png")
plt.show()
plt.figure(figsize=w, dpi=z)
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
bnipe21 = bnipe.reset_index()
print(bnipe)
bnipe2 = pd.DataFrame({"topics":nope["counts"],"comments":nope2["counts"]})
bnipe22 = bnipe2.reset_index()
print(bnipe2)
print(list)
print(wiwi)
df13 = pd.DataFrame({"date":wiwi})
try1 = bnipe2.loc["Вадим Карпусь","topics"]
try1 = pd.DataFrame(try1)
print(try1)
try2 = bnipe2.loc["Олег Данилов","topics"]
try2 = pd.DataFrame(try2)
print(try2)
df13  = pd.concat([df13,try1,try2])
#df13  = df13.merge(try2, left_index=True, right_index=True)
print(df13)

#df13 = df13.merge(try1)
for i in range(len(list)):
    ror = list[i]
    print(ror)
    df13[ror] = bnipe2.loc[ror,"topics"]
df13 = df13.fillna(0)
del df13["topics"]
#df13 = df13.iloc[len(wiwi):,:]    
print(df13)
length = len(wiwi)
df14 = df13.iloc[length:,:]
df14["date"] = wiwi
df14 = df14.reset_index(drop=True)
df14.sort_values(by="date")
print(df14)
#print(mimi)
#print(mipi)
#print(nopi)
#print(bnipe["ouy"])
#print(bnipe["author"])
def tracing(list):
    l1 = []
    l2 = []
    s1 = [*df14["date"]]
    s2 = [*df14[list]]
    for i in range(len(s2)):
        if s2[i] > 0:
            l1.append(s1[i])   
            l2.append(s2[i])
    trace = go.Bar(
    x=df14["date"],
    y=df14[list],
    name=list,
    text = s2,
    textposition = 'auto',
    hoverinfo = "name"
#        marker=dict(
#            color='rgb(219, 64, 82, 0.7)',
#            line=dict(
#                color='rgba(219, 64, 82, 1.0)',
#                width=2,
#                )
#            )
    )
    del l1,l2,s1,s2
#        marker=dict(
#            color='rgb(219, 64, 82, 0.7)',
#            line=dict(
#                color='rgba(219, 64, 82, 1.0)',
#                width=2,
#                )
#            )
    return trace
data = []
for i in range(len(list)):
    trace = tracing(list[i])
    data.append(trace)

layout = go.Layout(
    title='How many articles published at each day',
    xaxis=dict(
        title='Days',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Number of topics',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=1.0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    bargap=0.15,
    bargroupgap=0.1,
#    title='Hot & cold records',
#    xaxis=dict(
#        title='Days',
#        tickfont=dict(
#            size=14,
#            color='rgb(107, 107, 107)'
#        )
#    ),
#    yaxis=dict(
#        title='Count',
#        titlefont=dict(
#            size=16,
#            color='rgb(107, 107, 107)'
#        ),
#        tickfont=dict(
#            size=14,
#            color='rgb(107, 107, 107)'
#        )
#    ),
#    legend=dict(
#        x=0,
#        y=1.0,
#        bgcolor='rgba(255, 255, 255, 0)',
#        bordercolor='rgba(255, 255, 255, 0)'
#    ),
    barmode='stack'
#    bargap=0.15,
#    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='Summtable.html', auto_open = False)

list2 = ["1AM","2AM","3AM","4AM","5AM","6AM","7AM","8AM","9AM","10AM","11AM", \
         "12AM","1PM","2PM","3PM","4PM","5PM","6PM","7PM","8PM","9PM","10PM", \
         "11PM","12PM"]
list3 = ["00","01","02","03","04","05","06","07","08","09","10","11","12", \
         "13","14","15","16","17","18","19","20","21","22","23"]
print(len(list2))
df15 = df[["author","counts","wow"]]
#df15["wow"] = pd.to_datetime(df15["wow"])
df15 = df15.sort_values(by="wow")
df15 = df15.reset_index()
df15["topics"] = 1
df15["wow2"] = df15["wow"].str.split(":")
df15["wow2"] = df15["wow2"].str[0]
del df15["index"]
#df15.sort("wow")
print(df15)
print(df15.info())

nype = df.groupby(["author","wow"]).count()
nype2 = df.groupby(["author","wow"]).sum()
bnype = df.groupby(["wow","author"]).count()
bnype2 = df.groupby(["wow","author"]).sum()
nip = df15.groupby("wow").count()
nop = df15.groupby("author").sum()
nip2 = df15.groupby("author").count()
nop2 = df15.groupby("wow").sum()
print(df15.info())
nyp = df15.groupby(["author","topics"]).sum()
nyp2 = df15.groupby(["author","topics"]).count()
nap = df15.groupby("topics").count()
nap2 = df15.groupby("topics").sum()
nup = df15.groupby(["author","wow2"]).count()
nup2 = df15.groupby(["author","wow2"]).sum()
nep = df15.groupby(["wow2","author"]).count()
nep2 = df15.groupby(["wow2","author"]).sum()
nep = nep.reset_index()
nep2 = nep2.reset_index()
#print(nup,nup2)
print(nep,nep2)
#print(nip,nop,nip2,nop2,nap,nap2,nyp,nyp2, bnype2)
bnype2 = bnype2.reset_index(drop=True)
#print(bnype2.dtypes)
#print(bnype2.info())

nop = nop.reset_index()
nop["authors"] = list
print(nop)
#for i in range(len(list3)):
#
#    borgi = nep2.loc[nep["wow2"]==i,"topics"]
#    nop[list3[i]] = borgi
print(nop)
nep2 = (nep2.pivot('author','wow2','topics')
        .fillna(0)
        .reindex(columns=list3, fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1))
print(nep2)
#df = pd.DataFrame({"hours":list3})
#nep2 = (nep2.pivot('author','wow2','topics')
#        .fillna(0)
#        .reindex(columns=list, fill_value=0)
#        .reset_index()
#        .rename_axis(None, axis=1))
df = nep2.transpose()
df = df.iloc[1:]
df.columns = list

print(df)
#print(bnype2.describe())
#try1 = nep2.loc["07","author"]
#try1 = pd.DataFrame(try1)
#print(try1)
#try2 = nep2.loc["08","author"]
#try2 = pd.DataFrame(try2)
#print(try2)
#df16  = pd.concat([df13,try1,try2])
##df13  = df13.merge(try2, left_index=True, right_index=True)
#print(df16)
'''
import pandas as pd
df = pd.DataFrame({"author":["A","B", "B","C","A","C"],
                   "hour":["h01","h02","h04","h04","h05","h05"],
                   "number_of_topics":["1","4","2","6","8","3"]})
print(df)
df2 = pd.DataFrame({"author":["A","B","C"],
                    "h01":["1","0","0"],
                    "h02":["0","4","0"],
                    "h03":["0","0","0"],
                    "h04":["0","2","6"],
                    "h05":["8","0","3"],
                    "h06":["0","0","0"]})
print(df)
print(df2)
cols = ['h{:02d}'.format(x) for x in range(1, 7)]
df = (df.pivot('author','hour','number_of_topics')
        .fillna(0)
        .reindex(columns=cols, fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1))
print (df)
'''
#
##df13 = df13.merge(try1)
#for i in range(len(list)):
#    ror = list[i]
#    print(ror)
#    df13[ror] = bnipe2.loc[ror,"topics"]
#df13 = df13.fillna(0)
#del df13["topics"]
##df13 = df13.iloc[len(wiwi):,:]    
#print(df13)
#print(df15)

df = df.reset_index(drop=True)
df["hour"] = list3
print(df)
print(df.dtypes)
print(df.info())
def tracing2(list):
    l1 = []
    l2 = []
    s1 = [*df["hour"]]
    s2 = [*df[list]]
    for i in range(len(s2)):
        if s2[i] > 0:
            l1.append(s1[i])   
            l2.append(s2[i])
    trace = go.Bar(
        x=df["hour"],
        y=df[list],
        name=list,
        text = s2,
        textposition = 'auto',
        hoverinfo = "name"
#        marker=dict(
#            color='rgb(219, 64, 82, 0.7)',
#            line=dict(
#                color='rgba(219, 64, 82, 1.0)',
#                width=2,
#                )
#            )
        )
    del l1,l2,s1,s2
    return trace
data = []
for i in range(len(list)):
    trace = tracing2(list[i])
    data.append(trace)

layout = go.Layout(
     title='When each author published an article?',
    xaxis=dict(
        title='Hours',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Number of topics',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    bargap=0.15,
    bargroupgap=0.1,
#    title='Hot & cold records',
#    xaxis=dict(
#        title='Days',
#        tickfont=dict(
#            size=14,
#            color='rgb(107, 107, 107)'
#        )
#    ),
#    yaxis=dict(
#        title='Count',
#        titlefont=dict(
#            size=16,
#            color='rgb(107, 107, 107)'
#        ),
#        tickfont=dict(
#            size=14,
#            color='rgb(107, 107, 107)'
#        )
#    ),
#    legend=dict(
#        x=0,
#        y=1.0,
#        bgcolor='rgba(255, 255, 255, 0)',
#        bordercolor='rgba(255, 255, 255, 0)'
#    ),
    barmode='stack'
#    bargap=0.15,
#    bargroupgap=0.1
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='Summtable2.html', auto_open = False)
df = df0
print(df.head())
print(datetime.datetime.today().weekday())
from datetime import date
import calendar
my_date = date.today()
woo = calendar.day_name[my_date.weekday()]
print(woo)
dt = '18.10.2001'
day, month, year = (int(x) for x in dt.split('.'))    
answer = datetime.date(year, month, day).today()
answer = calendar.day_name[answer.weekday()]
df["date"] = df["ouy"]
df['dow'] = pd.to_datetime(df['ouy'], format='%d.%m.%Y').dt.weekday_name
df = df[["author","counts","date","dow"]]
print(answer)
df = df.sort_values(by="dow")
df = df.reset_index(drop=True)
#df["ouy3"] = df["ouy"].today()
#df["ouy4"] = calendar.day_name[df["ouy"].weekday()]
print(df.head())
list = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
nipi = df.groupby(["dow","date"]).count().reset_index()
nipi2 = df.groupby(["dow","date"]).sum().reset_index()
nipi3 = pd.DataFrame({"dow":nipi["dow"], "date":nipi["date"], \
                      "topics":nipi["counts"], "comments":nipi2["counts"]})
nipi3 = nipi3.sort_values(by="dow")
nopi = df.groupby(["date","dow"]).count().reset_index()
nopi2 = df.groupby(["date","dow"]).sum().reset_index()
nopi3 = pd.DataFrame({"dow":nopi["dow"], "date":nopi["date"], \
                      "topics":nopi["counts"], "comments":nopi2["counts"]})
nopi3 = nopi3.sort_values(by="dow")
nopi3 = round(nopi3.groupby("dow").sum()/nopi3.groupby("dow").count(),0)
#nopi3 = nopi3.reset_index()
del nopi3["date"]
nopi3.rename(columns={'comments':'avgcomm',"topics":"avgtop"}, inplace=True)
nopi3["avgcomm"] = nopi3["avgcomm"].astype(int)
nopi3["avgtop"] = nopi3["avgtop"].astype(int)
#nopi3['dow'] = pd.Categorical(nopi3['dow'], categories=list, ordered=True)
#nopi3["dow"] = nopi3["dow"].sort_index()
#crashes_by_day = nopi3["dow"].value_counts()
#crashes_by_day = crashes_by_day.sort_index()
#nopi3["dow"] = crashes_by_day.index
sorterIndex = dict(zip(list,range(len(list))))
nopi3['dow'] = nopi3.index
nopi3['dow'] = nopi3['dow'].map(sorterIndex)
print(nopi3.head())
nopi3.sort_values('dow', inplace=True)
del nopi3["dow"]
nopi3 = nopi3.reset_index()
nopi3["avgavg"] = round(nopi3["avgcomm"]/nopi3["avgtop"])
nopi3["avgavg"] = nopi3["avgavg"].astype(int)
print(nopi3.head())
print(nipi)
print(nipi2)
print(nopi)
print(nopi2)
print(nipi3)
print(nopi3)
#print(crashes_by_day)
data = [go.Bar(x=nopi3["dow"],
               y=nopi3["avgcomm"],
               text=nopi3["avgcomm"],
               textposition = 'auto',
               hovertext = "text",
               marker=dict(
                   color='rgb(158,202,225)',
                   line=dict(
                        color='rgb(8,48,107)',
                        width=1.5),
                    ),
                opacity=0.6
            
)]
layout = go.Layout(
    title='Avgcomm',
    xaxis=dict(
        title='Day of week',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Number of comments',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    bargap=0.15,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='Avgcomm.html', auto_open = False)
data = [go.Bar(
            x=nopi3["dow"],
            y=nopi3["avgtop"],
            text=nopi3["avgtop"],
            textposition = 'auto',
            hovertext = "text",
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                     color='rgb(8,48,107)',
                     width=1.5),
                 ),
            opacity=0.6
            
            
)]
layout = go.Layout(
    title='Avgtop',
    xaxis=dict(
        title='Day of week',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Number of topics',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    bargap=0.15,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='Avgtopics.html', auto_open = False)
data = [go.Bar(
            x=nopi3["dow"],
            y=nopi3["avgavg"],
            text=nopi3["avgavg"],
              textposition = 'auto',
              hovertext = "text",
              marker=dict(
                  color='rgb(158,202,225)',
                  line=dict(
                       color='rgb(8,48,107)',
                       width=1.5),
                   ),
               opacity=0.6
            
            
)]
layout = go.Layout(
    title='Avgavg',
    xaxis=dict(
        title='Day of week',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Average of comments per topic',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    bargap=0.15,
    bargroupgap=0.1
)
fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='Avgavg.html', auto_open = False)
print(df14)
#pi = df.groupby("date").count()
#print(pi)
#pi2 = df.groupby("date").sum()
#print(pi2)
#pi3 = [*np.unique(df["date"])]
#print(pi3)
bnipe.to_csv("summarytable1.csv")
bnipe2.to_csv("summarytable2.csv")
k.to_csv("authorsanalyze.csv")