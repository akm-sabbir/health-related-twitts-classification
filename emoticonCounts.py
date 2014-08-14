#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model.base import LinearModel
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import scipy as sp
import numpy as np
from sklearn import preprocessing
from sklearn import svm
import nltk
import re
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
from nltk import RegexpTokenizer

polarity = {'positive','negative','neutral'}
def dict_parser(data_source):
    for range_elem in xrange(0,len(data_source)):
        data_source[range_elem] = data_source[range_elem].replace('"','')
        data_source[range_elem] = data_source[range_elem].replace('.',' ')
        #data_source[range_elem] = data_source[range_elem].replace('\'','')
    dict_vec = DictVectorizer()
    mat = dict_vec.fit_transform(Counter(s.split()) for s in data_source)
    arr_dict = dict_vec.inverse_transform(mat)
    
    return arr_dict
def emoticon_count(path_name='C:\\Binary_classifier_for_health\\DataSet_2_supriya.txt',emoticon_file = 'C:\\Binary_classifier_for_health\\EmoticonLookupTable.txt',\
                   emoticon_granual = 'C:\\Binary_classifier_for_health\\emoticonsWithPolarity.txt'):
    global polarity
    pat1 = '\(\^'
    pat2 = ' \^\)'
    file_to_read = open(path_name)
    vectorizer = CountVectorizer(ngram_range=(1,2))
    emoticon_vec = {}
    emoticon_granual_vec = {}
    try:
        data = file_to_read.readlines()
    finally:
        file_to_read.close()
    arr_dict = dict_parser(data)
    file_to_read_emoticons = open(emoticon_file)
    try:
        emoticon_data = file_to_read_emoticons.readlines()
    finally:
        file_to_read_emoticons.close()
    file_to_read_emoticon_granual = open(emoticon_granual)
    try:
        emoticon_granual_data = file_to_read_emoticon_granual.readlines()
    finally:
        file_to_read_emoticon_granual.close()
    for datum in emoticon_granual_data:
        sub_data = datum.split('\t')
        type_data = sub_data[1]
        ob = re.search(pat1 + '(?=' + pat2+')', sub_data[0], re.I)
        if(ob != None):
            print('pattern is: ' + pat1 + pat2)
            sub_sub_data = sub_data[0].split('(^ ^)')
            emoticon_granual_vec['(^ ^)'] = sub_data[1]
            sub_data[0] = sub_sub_data[0] +' '+sub_sub_data[1]
        sub_sub_data = sub_data[0].split(' ')
        for each in sub_sub_data:
            if(emoticon_granual_vec.get(each,None) == None):
                emoticon_granual_vec[each] = sub_data[1] 
        
    X = []
    for datum in data:
        sub_data = datum.split('\t')
        X.append(sub_data[3].decode('utf-8','ignore'))
    matrix = vectorizer.fit_transform(X)
    reverse_arr = vectorizer.inverse_transform(matrix)
    for datum in emoticon_data:
        sub_datum = datum.split('\t')
        if(emoticon_vec.get(sub_datum[0],None)== None):
            emoticon_vec[sub_datum[0]] = sub_datum[1]
    user = {}
    print('lenght of X is: ' + str(len(X)))
    print('lenght of arr_dict is: ' + str(len(arr_dict)))
    h_arr = np.zeros((len(arr_dict),3))
    pos,neg,neu = 0,0,0
    counter_map = dict(emoticon_vec.items() + emoticon_granual_vec.items())
    for i in xrange(0,len(arr_dict)):
        '''print('doc #' + str(i)+' data: ')
        print( arr_dict[i])
        print('\n')'''
        for keys in arr_dict[i]:
            if(counter_map.get(keys,None) != None):#emoticon_vec.get(keys,None) != None or emoticon_granual_vec.get(keys,None) != None):
                #print('reverse doc pos and element:'+ str(i) +' '+ keys)
                if(counter_map[keys].lower().find('positive') != -1):
                    h_arr[i][0] = 1
                    print('indexes :' + str(i) +' '+ str(0))
                    pos += 1
                elif(counter_map[keys].lower().find('negative') != -1):
                    print('indexes :' + str(i) +' '+ str(1))
                    h_arr[i][1] = 1
                    neg += 1
                else:
                    print('indexes :' + str(i) +' '+ str(2))
                    h_arr[i][2] = 1
                    neu += 1
        if(user.get(i,None) == None):
            user[i] = 0
    count = 0
    for elem in xrange(0,len(user)):
        if(user[elem] != 0):
            count += 1
    print("number of elements have the emoticons: " + str(count)+' '+str(pos) +' ' + str(neg) +' '+str(neu))
    return h_arr
emoticon_count()