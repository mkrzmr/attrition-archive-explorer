#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 20:17:29 2021

@author: michael
"""
# first load all files, use index.html if no mirror.html is around

import glob
import pandas as pd
import pandas_explode
pandas_explode.patch()
from IPython.display import display
from bs4 import BeautifulSoup
import re, datetime
import os
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import punkt
import spacy
from tld import get_tld
import networkx as nx
from pyvis.network import Network
import numpy as np

nlp = spacy.load('en_core_web_sm')
nlp.max_length = 3000000

sw = stopwords.words('english') #defining stopwords and removing them before plotting # stopwords need to be edited
add_stopwords = ['domains', 'mirrormultiple', 'mirror.html']
for w in add_stopwords:
    if w.lower() not in sw:
        sw.append(w)



files = []

def write_dataframe():
    progress = 0
    for i in glob.glob('/home/michael/Desktop/attrition analysis/****/**/**/*/*.html', recursive = False): #reduce scope for testing, first make list
        if i[-11:] == 'mirror.html' and os.path.isfile(i) == True:
            with open(i, 'rb') as infile:
                url = re.search("w{3}\.\S+\/", i)
                if url is not None:
                    tld = get_tld(url.group(), fail_silently=True, fix_protocol=True) #extracting tlds
                    url = url.group()
                else:
                    tld = "No url extracted"
                #print(url.group())
                soup = BeautifulSoup(infile, "html5lib")
                findurls = soup.find_all('a') #find all links
                linklist =[] #create empty linklist
                for tag in findurls:
                    link = tag.get('href',None) #extract link targets
                    if link is not None:
                        linklist.append(link)
                clean = soup.get_text(strip=True) #strip away html tags
                clean = clean.encode()
                match = re.search('\d{4}/\d{2}/\d{2}', i)
                date = datetime.datetime.strptime(match.group(), '%Y/%m/%d').date() # get dates from folder structure IDEA: extract domain like that #done
            files.append((i, date, url, tld, clean, linklist))
            progress +=1
            print('Processed '+ str(progress) + ' pages.')
        elif i[-10:] == 'index.html' and os.path.isfile(i) == True:
            with open(i, 'rb') as infile:
                url = re.search("w{3}\.\S+\/", i)  
                if url is not None:
                    tld = get_tld(url.group(), fail_silently=True, fix_protocol=True) #extracting tlds
                    url = url.group()
                else:
                    tld = "No url extracted"
                soup = BeautifulSoup(infile, "html5lib")
                linklist =[] #create empty linklist
                findurls = soup.find_all('a') #find all links
                for tag in findurls:
                    link = tag.get('href',None) #extract link targets
                    if link is not None:
                        linklist.append(link)
                clean = soup.get_text(strip=True) #strip away html tags
                clean = clean.encode()
                match = re.search('\d{4}/\d{2}/\d{2}', i)
                date = datetime.datetime.strptime(match.group(), '%Y/%m/%d').date() # get dates from folder structure
            files.append((i, date, url, tld, clean, linklist))
            progress +=1
            print('Processed '+ str(progress) + ' pages.')
            
    df = pd.DataFrame(files) #turn everything into Dataframe
    df.to_csv('file.csv') #next try and read from file
    display(df)

def read_dataframe():
    df = pd.read_csv("file.csv")
    print(df(20))

def defacements_over_time():
    df = pd.read_csv("file.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    #result = df.groupby(df['Date'].map(lambda x: x.year)).count() #group data by year and month
    result = (df.groupby([df['Date'].dt.year.rename('year'), df['Date'].dt.month_name().rename('month')]).count().reset_index()) #finally
    display(result)
    result.to_csv('distribution_time.csv', index = True) 

def frequency(stopwords): #runs frequency analysis on the whole corpus
    df = pd.read_csv("file.csv")
    clean_text=df.Text.to_string() #convert to string
    #print(clean_text)
    token = nltk.word_tokenize(clean_text) #Simple tokenization
    token2 = list(filter(lambda token : punkt.PunktToken(token).is_non_punct,token)) #Remove punctuation 
    token3 = [word.lower() for word in token2] #all lowercase
    token4 = [word for word in token3 if word not in stopwords] #filter out stopwords #need to give this a special set of stopwords
    clean_token = token4
    freq = nltk.FreqDist(clean_token)
    #plt.figure(figsize=(20, 8))#use this to change plot size
    #freq.plot(20, cumulative=False) #plot a graph # nice for preview, probably better to save all data to file
    #freq.to_csv('tokencount_total.csv', index = True)
    
def bigrams(stopwords): #runs bigram analysis on the whole corpus
    df = pd.read_csv("file.csv")
    clean_text=df.Text.to_string() #convert to string
    token = nltk.word_tokenize(clean_text) #Simple tokenization
    token2 = list(filter(lambda token : punkt.PunktToken(token).is_non_punct,token)) #Remove punctuation 
    token3 = [word.lower() for word in token2] #all lowercase
    token4 = [word for word in token3 if word not in stopwords] #filter out stopwords
    clean_token = token4
    bigrams = list(nltk.bigrams(clean_token))
    freq_bi = nltk.FreqDist(bigrams)
    frame = pd.DataFrame({'bigrams':freq_bi}) #convert the list into a dataframe
    #print(frame.head(100)) #testing
    frame.to_csv('bigrams.csv', index=True, header=True)
    freq_bi.most_common(20)
    for i in freq_bi.most_common(20):
        print(i[0], i[1])
    plt.figure(figsize=(20, 8))#use this to change plot size
    freq_bi.plot(20)
    
    
def tld_over_time():
    df = pd.read_csv("file.csv")
    result = df.groupby('Tld').size()
    result.plot(kind = 'bar', figsize=(50,30), linewidth=10, alpha=1, stacked = False, table = False)
    plt.xlabel('TLD') #xlabel
    plt.ylabel('Total') #ylabel
    result.to_csv('tld_total.csv', index = True) 
    plt.show()
    
def nlp_entlist():
    df = pd.read_csv("file.csv")
    entlist = []
    for i, row in df.iterrows():
        #print(f"Index: {i}")
        #print(f"{row['Text']}")
        page = nlp(f"{row['Text']}")
        for ent in page.ents:
            entlist.append([ent.text, ent.label_, i])
        print("Processed "+str(i)+" Pages.")
    entlist = pd.DataFrame(entlist)
    entlist.to_csv('nlp_entlist.csv', index = True)
        
       # page = nlp(row['Text'].to_string())
        #for i in page.ents:
         #print(i.text, i.label_)
             #entlist.append((i.text, i.label_, row.No))
    # for i in page.ents:
    #     #print(i.text, i.label_)
    #     entlist.append((i.text, i.label_, df.No))
    # entlist = pd.DataFrame(entlist)
    #entlist.to_csv('nlp_entlist.csv', index = True)
    
def pivot_entlist(): #runs some basic analysis on the results
    df=pd.read_csv("nlp_entlist.csv")
    output = df['ent_text'].value_counts() #first time I find pandas more helpful and straightforward than gui solutions
    output.to_csv('enttext_count.csv')
    output = output = df['ent_type'].value_counts()
    output.to_csv('enttype_count.csv')
    
def link_network(): #lets go
    df = pd.read_csv("file.csv")
    rows = list()

    for row in df[['Url', 'Links']].iterrows():
        r = row[1]
        for i in r.Links.split(): #remove all stopwords
            if i.lower() not in sw:
                i =i.strip('[ ]')
                rows.append((r.Url, i))
    linknet = pd.DataFrame(rows, columns=['Url', 'Links'])   
    linknet.replace("'mirror.html'", np.nan, inplace=True) #removing mirror.html
    linknet.replace("'index.html'", np.nan, inplace=True) #removing mirror.html
    linknet.replace('', np.nan, inplace=True)    #replace empty cells with NaN
    linknet.dropna(inplace=True)                #remove NaN

    #display(linknet.head(100))
    linknet.to_csv('linknet.csv') #save to csv
    ### confige visualization ####
    G = nx.from_pandas_edgelist(linknet.head(2000), #It can run on the full set, but is very slow
                                    source ='Url',
                                    target = 'Links')
    #nx.draw_circular(G)
    print(nx.info(G)) #show numbers of graphs and edges
    net = Network(height='2000px', width='100%', bgcolor='#222222', font_color='white') #size and color
    net.show_buttons(filter_ = ['nodes']) #show options in html
    nx.write_gexf(G, 'attrition.gexf') #write graph info to file

    net.from_nx(G)
    net.show('example.html')
    
    
    
#write_dataframe()
#defacements_over_time()
#frequency(sw)
#bigrams(sw)
#tld_over_time()
#nlp_entlist()
#pivot_entlist()
link_network()