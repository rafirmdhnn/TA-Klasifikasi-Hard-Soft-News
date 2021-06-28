# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 14:40:48 2021

@author: rafir
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import string
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

class dataPreparation(object):
    
    def __init__(self, data):
        self.data = data
    
    def labeling(self):
        self.num_label= []
        for i in range(0, len(self.data)):
            self.label = self.data['label']
            self.label_idx = self.label[i]
            #mengubah label kedalam bentuk numeric "0" dan "1"
            if self.label_idx == "Soft News":
                self.num_label.append("1")
            else:
                self.num_label.append("0")
        return self.num_label
    
    def preProc(self):
        self.corpus = []
        self.factory_stem = StemmerFactory()
        self.stemmer = self.factory_stem.create_stemmer()
        self.indo = stopwords.words('indonesian')
        isi = self.data["isi"]
        
        #Membaca dan menjalakan preprocessing pada setiap dokumen berita
        
        for i in range(0, len(self.data)):
            self.isi = isi[i]
            
            #caseFolding atau mengubah huruf kapital menjadi lowercase
            self.lower = self.isi.lower()
            
            #remove angka
            self.remAngka = re.sub('[^a-zA-Z]+', ' ', self.lower)
            
            #remove punctuation 
            self.remPunct = self.remAngka.translate(str.maketrans("","", string.punctuation))
            
            #remove whitespace pada awal dan akhir kata/kalimat
            self.remSpace = self.remPunct.strip()
            
            #melakukan pemecahan kalimat menjadi token
            self.token = nltk.tokenize.word_tokenize(self.remSpace)
            
            #remove stopwords dan mengubah kata kedalam bentuk dasar (Steming)
            self.ss_word = [self.stemmer.stem(word) for word in self.token if not word in self.indo]
            
            self.join_word = ' '.join(self.ss_word)
            self.corpus.append(self.join_word)
            
        return self.corpus
    
    def BoW(self):
        terms = self.corpus
        
        #Ubah list menjadi kedalam bentuk string
        self.listToStr = ' '.join(str(x) for x in terms)
        #split string menjadi kata per kata 
        self.words = self.listToStr.split(' ')
        #hitung frekuensi dari setiap kata yang ada
        self.count = Counter(self.words)
        
        #plotting chart untuk melihat keseluruhan frekuensi kata
        freqTerms = pd.DataFrame(self.count.most_common(50),
                                  columns=['words', 'count'])
        freqTerms.head()
        fig, ax = plt.subplots(figsize=(8, 8))
        freqTerms.sort_values(by='count').plot.barh(x='words', y='count', ax=ax, color="blue")
        
        ax.set_title("Kata yang Sering Muncul pada Artikel Hard News")
        plt.show()
        
        #plotting wordcloud
        wordcloud = WordCloud(max_font_size=50, max_words=800, background_color="white").generate(self.listToStr)                     
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.tight_layout(pad = 0)
          
        plt.show()
        
        return self.count
        
    def tfidf(self, fitur):
        self.tfidf_vect = TfidfVectorizer(analyzer='word', use_idf=True, smooth_idf=True ,vocabulary=(fitur))
        self.tfidf_query = self.tfidf_vect.fit_transform(self.corpus)
            
        self.tfidf_query_df = pd.DataFrame(data = self.tfidf_query.toarray(), index = self.data["dokumen"])
        self.tfidf_query_df.columns = self.tfidf_vect.get_feature_names()
        self.tfidf_query_df.head()
        self.tfidf_query_df["label"] = self.num_label  
    
    def saveCsv(self, outPre, sglDtst):
        self.outPre = outPre
        self.singleDataset = sglDtst
        #output preProcessing to csv
        preProc_df = pd.DataFrame(data = self.corpus, index = self.data["dokumen"])
        preProc_df.columns = ['output']
        preProc_df.to_csv(self.outPre)
        
        #output tf-idf to csv
        self.tfidf_query_df.to_csv(self.singleDataset)
#%%
#import data csv
file = open(r'C:\\Skripsi\Skripsi\Sidang\Dataset\raw.csv','r')
dataset = pd.read_csv(file, delimiter = ';',  encoding='cp1252' )
isi = dataset['isi']

#initialize object
preProcessing = dataPreparation(dataset)
#preprocessing
out = preProcessing.preProc()
#labeling into numeric
label = preProcessing.labeling()
#bag-of-word
bow = preProcessing.BoW()

#%%

#inisiasi fitur untuk input key word tf-idf
fitur = ['unesco',
         'gempa',
         'menteri',
         'rusak',
         'longsor',
         'banjir',
         'perintah',
         'ekonomi',
         'pidana',
         'pemprov',
         'presiden', 
         'camat',
         'korban',
         'preman',
         'polisi',
         'pemkab',
         'bupati',
         'kebakaran',
         'gubernur',
         'infrastruktur',
         'pergub',
         'kerjasama',
         'perda',
         'kabupaten',
         'macet',
         'sampah',
         'tsunami',
         'kampanye',
         'pemilu']

#hitung bobot dgn tfidf
out_tfidf = preProcessing.tfidf(fitur)

#save file kedalam csv format
pathOutPre = r'C:\Skripsi\Skripsi\Sidang\Dataset\outputPreProcessing.csv'
pathSglDt = r'C:\Skripsi\Skripsi\Sidang\Dataset\single-dataset.csv'

preProcessing.saveCsv(pathOutPre, pathSglDt)