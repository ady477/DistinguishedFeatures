# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:30:33 2016

@author: adityat
"""
from __future__ import division

import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from random import shuffle
import timeit
from sklearn.decomposition import PCA
from sklearn import decomposition
import sys

import nltk, re, pprint
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *

def removestopwords(text, stopwords):
    pass

#### MAIN STARTS HERE ############
if __name__ == "__main__":
    proteins = []
    classes = []
    Abstract = []
    
    #Extract Data
    proteins = pickle.load(open('outputPKL/proteins.pkl', 'rb'))
    print "len(proteins) : ",len(proteins)
    proteins = [x for x in proteins if x is not None]
    print "len(proteins) : ",len(proteins)
    classes = pickle.load(open('outputPKL/Category_GO_ID.pkl', 'rb'))
    classes = [x for x in classes if x is not None]
    Abstract = pickle.load(open('outputPKL/Abstract.pkl', 'rb'))
    Abstract = [x for x in Abstract if x is not None]
    Abstract_new = []
    
    for i in range(0,len(Abstract)):
        Abstract[i] = list(set(Abstract[i]))
        Abstract[i] = ' '.join(Abstract[i])
        
    #Stop words from NLTK
    cachedStopWords = set(stopwords.words('english'))
    #Porter stemmer
    stemmer = PorterStemmer()
    # Pre Process Data    
    for i in range(len(Abstract)):
        my_text = Abstract[i]
        my_text = my_text.lower()
        my_text = re.sub(r'([^\s\w]|_)+', '', my_text)
        my_text = ' '.join([word for word in my_text.split() if word not in cachedStopWords])
        stemmed = []
        for word in my_text.split():
            stemmed.append(stemmer.stem(word))
        stemmed = ' '.join(stemmed)
        vectorizer = CountVectorizer(ngram_range=(1,2))
        analyzer = vectorizer.build_analyzer()
        Abstract_new.append(analyzer(stemmed))
    
    pickle.dump('outputPKL/Abstract_new.pkl', 'wb')