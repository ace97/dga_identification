import pandas as pd


df=pd.read_csv("dgadom.csv")
print df.shape
for i in range (133926):
    if df['class'][i]=='legit':
        df['class'][i]=0
    else:
        df['class'][i]=1

del df['domain']
subclass = {'legit': 0,'newgoz':1,'goz':2,'cryptolocker':3 }

df.subclass = [subclass[item] for item in df.subclass]
df=df[['host','subclass','class']]
print df.head()

df.to_csv('mix.csv')
