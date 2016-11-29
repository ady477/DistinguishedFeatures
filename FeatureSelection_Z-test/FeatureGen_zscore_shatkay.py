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
from collections import defaultdict

# global constants
THRESHOLD = 1.150
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
    Abstract = pickle.load(open('outputPKL/Abstract_new.pkl', 'rb'))
    print "Prob. Calc."
    
    # Calculation of P(t/c) on the basis of Count(t,c), Count(DC)
    count_tc = defaultdict(float)
    count_dc = defaultdict(float)
    Vocab = defaultdict(bool)
    Classes = set()
    for d in range(len(Abstract)):
        temp_vocab = {}
        for t in Abstract[d]:
            if t not in temp_vocab:
                for c in classes[d]:
                    count_tc[t,c] += 1
                temp_vocab[t] = True
            if t not in Vocab:
                Vocab[t] = 1
        for c in classes[d]:
            count_dc[c] += 1
            Classes.add(c)
    
    Classes = list(Classes)
    
    for (t,c) in count_tc:
        count_tc[t,c] = count_tc[t,c]/count_dc[c]
    print "Prob. Calc. Done"
    
    #Z-score calculation
    print "Z-score Calc."
    Z = defaultdict(float)
    for t in Vocab:
        for c1 in Classes:
            for c2 in Classes:
                if c1 != c2 :
                    if (t,c1,c2) not in Z:
                        P_bar = ((count_dc[c1]*count_tc[t,c1]) + (count_dc[c2]*count_tc[t,c2]))/(count_dc[c1] + count_dc[c2])
                        den = math.sqrt(P_bar*(1-P_bar)*((1/count_dc[c1]) + (1/count_dc[c2])))
                        if den != 0:
                            Z[t,c1,c2] = (count_tc[t,c1] - count_tc[t,c2])/(den)
                            Z[t,c2,c1] = -Z[t,c1,c2]
    print "Z-score Calc. Done"
    
    # Threshold based distinguishing term
    print "Calc. Distinguishing terms based on Z-score"
    Distinguish_Features = set()
    for t in Vocab:
        for c1 in Classes:
            for c2 in Classes:
                if c1 != c2 :
                    if (t,c2,c1) in Z:
                        if abs(Z[t,c2,c1]) >= THRESHOLD:
                            Distinguish_Features.add(t)
    print "Calc. Distinguishing terms based on Z-score DONE"
                            
                    
            