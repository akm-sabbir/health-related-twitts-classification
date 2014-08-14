#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams, trigrams
import math
import numpy as np
import scipy as sp
from sklearn.utils import shuffle
import scipy.sparse as sps
from warnings import catch_warnings
from scipy.sparse import *
from scipy import *
from sklearn import svm, cross_validation
from sklearn import linear_model
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model.coordinate_descent import LinearModelCV
from sklearn import metrics
from sklearn.metrics import classification_report
import pickle
import random
from scipyp.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
import operator
from sklearn import svm
import pickle
from sklearn import preprocessing
from scipy.stats.distributions import logistic
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import precision_score
import sklearn.preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import tldextract
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
import tokenizer
import subprocess
from tokenizer import TwitterCountVectorizer
import itertools
from multiprocessing import Pool,Lock,Manager
class result_collector():
	def __init__(self,P_ = 0,R_ = 0,F1_ = 0):
		self.P = P_
		self.R = R_
		self.F1 = F1_
		self.beta = 0.0
		self.best_combination = None
		return
class urlsComponent(object):
	def __init__(self,domain_=None,subDomain_=None,suffix_=None,list_data_ = None):
		self.domain = domain_
		self.subdomain = subDomain_
		self.suffix = suffix_
		self.list_data = list_data_
		return
def domain_extractor(url_name):
	urls = tldextract.extract(url_name)
	tempList = re.findall('[a-zA-Z]+',url_name)
	tempList = [tokens for tokens in tempList if len(tokens) > 3]
	if(tempList.count('http') != 0):
		tempList.remove('http')
	if(tempList.count(urls.domain)!= 0):
		tempList.remove(urls.domain)
	if(tempList.count(urls.subdomain)!= 0):
		tempList.remove(urls.subdomain)
	if(tempList.count(urls.suffix)!=0):
		tempList.remove(urls.suffix)
	url_object = urlsComponent(domain_ = urls.domain,subDomain_ = urls.subdomain,suffix_ = urls.suffix,list_data_ = tempList)
	return url_object
class fScore(object):
    p = 0
    r = 0
    f1 = {}
    def __init__(self,Tp_ = 0,Fp_ = 0, Tn_ = 0, Fn_ = 0,expecresultList_ = None,precresultList_ = None):
        self.Tp = Tp_
        self.Fp = Fp_
        self.Tn = Tn_
        self.Fn = Fn_
        self.expecresultList = expecresultList_
        self.preresultList = precresultList_
        return
class dataObjectContainer(object):
    def __init__(self,X= None,y=None,y_rank = None,latest_tweets = None,user_names=None):
        self.X = X
        self.y = y
        self.y_rank = y_rank
        self.latest_tweets = latest_tweets
        self.hash_tag = None
	self.user_name_list = user_names
        return
    
stopwords = nltk.corpus.stopwords.words('english')
dictUrls = {}
def storeModels(clf,i_i,t_t):
    if( t_t == 'c'):
        model_saver = open("TwitterBigData/linear_models/model_health"+str(i_i),"wb")
    else:
        model_saver = open("TwitterBigData/linear_models/vectorizer_health"+str(i_i),"wb")
    pickle.dump(clf, model_saver, pickle.HIGHEST_PROTOCOL)
    #model_saver.close()
    return
def generates_tf_idf_features(X,version):
	tf_idf_vectorizer = TfidfVectorizer(min_df = 0.1,stop_words = 'english')
	mat = tf_idf_vectorizer.fit_transform(X)
        mat =    Normalizer(copy=False).fit_transform(mat)
  	storeModels(clf = tf_idf_vectorizer,i_i=version,t_t='y')
  	return mat
def generates_features(X,version):
    vectorizer = CountVectorizer(min_df = 1,stop_words='english',ngram_range = (1,3))
    matrix = vectorizer.fit_transform(X)
    matrix = matrix.astype('double')
    #matrix = Normalizer(copy=False).fit_transform(matrix)
    # print(str(matrix.shape))
    # print(str(len(vectorizer.get_feature_names())))
    storeModels(clf = vectorizer, i_i =version, t_t = 'v')
    # print(str(matrix.shape))
    return matrix
def buildingClassifier(matrix,test_label,modelIndex):#,kf):
    #write_score = open("Binary_classifier_for_health/reportkfold.txt","w+")
    print('start of the training and testing function\n')
    class_label = np.array(test_label)
    #kf = cross_validation.KFold(len(class_label),5,indices = True)
    logRegression = linear_model.LogisticRegression()
    #svmRegression = svm.SVC(C = 1.0,kernel='rbf',degree = 3)#,class_weight = {1:0.2,0:0.8})#,class_weight={1:0.6, 0:0.4})#linear_model.LogisticRegression(C = 0.8)
    predictions = []
    expected = []
    list_of_result = []
    matrix = matrix.tocsr()
    print(' i am inside of the training and testig function\n')
    '''for train_index,test_index in kf:
        X_train,X_test = matrix[train_index],matrix[test_index]
        Y_train,Y_test = class_label[train_index],class_label[test_index]'''
    print('start operation')
    score =logRegression.fit(matrix, class_label)
        #print(str(score))
    '''list_of_result.append(fScore(expecresultList_ = Y_test,precresultList_ = logRegression.predict(X_test)))
    predictions.extend(logRegression.predict(X_test))
    expected.extend(Y_test)'''
        #write_score.write("%0.3f %0.3f"%classification_report(expected, predictions))
        #print("Accuracy %0.3f (+/- %0.3f)"%(score.mean(),score.std()*2))
    #print("micro score %.3f " % precision_score(expected,predictions,average = 'micro'))
    #print("macro score %0.3f " % precision_score(expected,predictions,average='macro'))
    score = logRegression.fit(matrix,class_label)
    storeModels(clf = logRegression,  i_i = modelIndex, t_t = 'c')
    #write_score.close()
    # logRegression.predict(matri)
    return list_of_result
def test_phase(test_data,test_label,hash_test,latest_tweets,user_name_list,test_topic_features,h_arr,username_arr,pos_tag_mat):
    clf = loadModels(1.5)
    vectorizer = load_vectorizer(1.5)
    vectorizer_for_urls = load_vectorizer(1.6)
    vectorizer_tf_idf = load_vectorizer(1.3)
    vectorizer_character_ngrams = load_vectorizer(1.7)
    write_score = open("Binary_classifier_for_health/report1.6.txt","w+")
    write_data = open("Binary_classifier_for_health/output_data1_predcit01.5.txt","w+")
    #print(str(len(vectorizer.get_feature_names())))
    test_mat = vectorizer.transform(test_data)
    #test_mat = pos_mat
    character_ngrams_mat = vectorizer_character_ngrams.transform(test_data)
    test_mat = test_mat.astype('float')
    test_mat = hstack([test_mat,test_topic_features])
    test_mat = hstack([test_mat,character_ngrams_mat])
    test_mat = hstack([test_mat,h_arr])
    test_mat = hstack([test_mat,pos_tag_mat])
    #test_mat =hstack([test_mat,username_arr])
    print(test_mat.shape)
    #test_mat = vectorizer_tf_idf.transform(test_data)
    #test_mat = Normalizer(copy=False).transform(test_mat)
    test_mat_for_urls = vectorizer_for_urls.transform(latest_tweets)
    # test_mat = hstack([test_mat,hash_test])
    #test_mat = hstack([test_mat,test_mat_for_urls])
    test_label = np.array(test_label)
    predict_ = clf.predict(test_mat)
    #predict_ = logRegression.decision_function(test_mat)
    list_of_users = predict_.flatten().tolist()
    #print('pos class: '+ str(num) +'neg class: '+ str(predict_.shape[0]- num))
    #predict_result =  logRegression.predict_proba(test_mat)
    for elem in range(0,len(test_label)):
    	if(predict_[elem] == 0 and test_label[elem] == 1):
		write_data.write(user_name_list[elem]+'\t'+test_data[elem]+'\n')
    write_data.close()
    write_score.write(classification_report(np.array(test_label), predict_))
    write_score.close()
    return
def read_data_source(path):
    file_to_read = open(path,'r+')
    try:
        data = file_to_read.readlines()
    finally:
        file_to_read.close()
    
    return data
def startLoadingUrls(range_i):
    urls_vocabulary = list()
    global dictUrls
    for i in range(0,range_i):
        file_object_urls = open('Binary_classifier_for_health/expandedUrls'+str(i)+'.txt')
  	try:
            data = file_object_urls.readlines()
        finally:
            file_object_urls.close()
        urls_vocabulary.extend(data)
        #print(len(urls_vocabulary))
    
    for dat in urls_vocabulary:
        stringVal = dat.split("\t")     
	# if(len(stringVal) == 5 or len(stringVal) == 4):
	#   stringVal[2].strip()
	if(dictUrls.get(stringVal[1],None)==None):
		dictUrls[stringVal[1]]= set()
        dictUrls[stringVal[1].strip()].add(stringVal[2].strip())
            #print(dictUrls[stringVal[2]])
	# else:
	#    dictUrls[stringVal[2]] = None;
    return
def dict_Of_Features(data):
    #global indexingDic
    global dictUrls
    #global normalizer
    #labelList
    strings = 'doc'
    #previously used the following line
    #file_object_read = open("C:\\TwitterBigData\\tweetMessage\\combinedLabelTrainedClassesW2.txt")
    '''file_object_read = open(dir)
    try:
        data = file_object_read.readlines()
    finally:
        file_object_read.close()'''
    i = 0
    j = 0
    dictionaryForDoc = {}
    # file_object_Towrite = open("C:\\TwitterBigData\\tweetMessage\\positiveUrls.txt","w");
    countIndicator = 0
    list_of_tweets = []
    for dat in range(0,len(data)):
       # tempData = data[dat].split("\t")
        tempSubData = data[dat].split(' ')#[dat][0].split(" ")
        create_twitt = ''
        #print(tempSubData)
        # the following line for labeling we dont need until we do supervised classification
       # labelList.append(tempData[3])
        #del tempData
        tempString = ""
        dictForFeatures = {}
        indicator = 0
        for tempSubDat in tempSubData:
            tempSubDat = tempSubDat.replace("\n","")
	    if (len(tempSubDat) == 0): continue
            if(tempSubDat[0]== "@"):
              #  print('hashTag#: ,' + tempSubDat)
                tempSubDat =  tempSubDat.replace("@","")
                if(dictForFeatures.get(1,0) == 0):
                    dictForFeatures[1]= set()
                dictForFeatures[1].add(tempSubDat) 
		create_twitt = create_twitt + tempSubDat +' '
            elif(tempSubDat[0]== "#"):
               # print('userMentions: ' + tempSubDat)
                tempSubDat = tempSubDat.replace("#","")
                if(dictForFeatures.get(2,0) == 0):
                    dictForFeatures[2] = set()
                if(indicator == 0):
                    indicator += 1
                    countIndicator += 1
                dictForFeatures[2].add(tempSubDat)
		create_twitt = create_twitt+ tempSubDat + ' '
            elif(re.search("(?P<url>https?://[^\s]+)", tempSubDat) != None):
               # print('urls are: ' + tempSubDat)
                if(dictForFeatures.get(3,0) == 0):
                    dictForFeatures[3] = list()
                tempSubDat = tempSubDat.strip()
		'''if(dictUrls.get(tempSubDat,None)!= None):'''
			#print('parsing the urls')
		url_object = domain_extractor(tempSubDat)#dictUrls.get(tempSubDat).strip())
		#print(str(url_object.list_data)+'\n')
                dictForFeatures[3].append(url_object.domain)#previuosly it was '' means nothing
		create_twitt = create_twitt + url_object.domain + ' '
		dictForFeatures[3].append(url_object.subdomain) # empty item should be checked
		create_twitt = create_twitt + url_object.subdomain + ' '
		dictForFeatures[3].append(url_object.suffix)
	        create_twitt = create_twitt + url_object.suffix + ' '
		for component in url_object.list_data:
			dictForFeatures[3].append(component.strip())
	                create_twitt = create_twitt + component.strip() + ' '
		# upto this point the if block's scope
                # if(dictUrls.get(tempSubDat,None) == None):
                    #file_object_Towrite.write(tempData[0]+"\t"+tempData[1]+"\t"+tempSubDat+"\n")
            else:
                if(dictForFeatures.get(4,0) == 0): # option for token format changing 
                    dictForFeatures[4] = list();
                #print('Simple text: ' + tempSubDat)
		#if(tempSubDat.lower().find('e-cig')!= -1 or tempSubDat.lower().find('e-j')!= -1or tempSubDat.lower().find('e-*')):
                	#tempSubDat = tempSubDat.replace('-','')
                if( tempSubDat not in stopwords):
                    dictForFeatures[4].append(tempSubDat)
		    create_twitt = create_twitt + tempSubDat + ' '
                    #print(tempSubDat)
                    tempString = tempString + tempSubDat+" "
            del tempSubDat
        del tempSubData
        tempString = tempString.split(" ")
        dictionaryForDoc[strings+str(i)] = dictForFeatures #,data[dat][1])
	list_of_tweets.append(create_twitt)
        i = i + 1
        #print('iteration number: '+str(i)+"\n")
        # for dat in range(0,len( dictionaryForDoc)):
        #print(dictionaryForDoc[strings+str(dat)])
    del data
    
    j = 1
    overAll = 0
    total = 0
    print('countIndicator: '+ str(countIndicator))
    global maximus
    return dictionaryForDoc,list_of_tweets

def loadModels(version):
    file_to_read = open("TwitterBigData/Models/model_health"+str(version),'rb')
    try:
        clf = pickle.load(file_to_read)
    except:
        clf = linear_model.LogisticRegression()
    return clf
def load_vectorizer(version):
    file_to_read = open("TwitterBigData/Models/vectorizer_health"+str(version),'rb')
    try:
        vectorizer = pickle.load(file_to_read)
    except:
        vectorizer = CountVectorizer(min_df = 1,stop_words = 'english',ngram_range = (1,3))
    #print(str(vectorizer))
    return vectorizer
def find_urls(username,tweets):
	url_list = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweets)
	return url_list
def read_urls_from_source(source_path):
	dict_of_urls = {}
	file_to_read_urls = open(source_path,'r+')
	try:
		data = file_to_read_urls.readlines()
	finally:
		file_to_read_urls.close()
	for datum in data:
		sub_datum = datum.split('\t')
		#sub_sub_datum = sub_datum[2].split(' ') 
		if(dict_of_urls.get(sub_datum[0],None) == None):
			dict_of_urls[sub_datum[0]] = set()
		#for each in sub_sub_datum:
		dict_of_urls[sub_datum[0]].add(sub_datum[4])
			
	return dict_of_urls
def parse_data(data):
    X = []
    y = []
    match_string = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    user_name_list = []
    count_pos = 0
    global dictUrls
    get_dict_urls = read_urls_from_source('data/expandedincompleteUrls0.txt')
    for datum in data:
        list_data = datum.split('\t')
	url_list = find_urls(list_data[1],list_data[3])
	#print('current position' + str(count_pos))     
	#list_data[3] = list_data[3].replace('rt','')
	data_list = list_data[3].split(' ')
	temp_data_list = data_list[:]
	if(len(url_list) != 0):
		for elem in url_list:
			temp_data_list = [s for s in temp_data_list if s not in elem]
	temp_data_holder = ' '.join(temp_data_list)
	url_list = []
	# X.append(temp_data_holder)#list_data[3].decode('utf-8','ignore'))
	if(get_dict_urls.get(list_data[1],None) != None):
		for element in get_dict_urls[list_data[1]]:
			temp_data_holder = temp_data_holder+' '+element
			url_list.append(element)
	else:
	 	if(dictUrls.get(list_data[1],None) != None):
			for each_elem in dictUrls[list_data[1]]:
				temp_data_holder = temp_data_holder +' '+ each_elem
				url_list.append(each_elem)
	i = 0
	temp_datas = list_data[3]
	for m in re.finditer(match_string,list_data[3]):
		if(i == len(url_list)): break
		temp_datas = re.sub(match_string,url_list[i],temp_datas)
		i += 1
	X.append(temp_datas.decode('utf-8','ignore'))#temp_data_holder.decode('utf-8','ignore'))
	#print(temp_data_holder)
	user_name_list.append(list_data[1].decode('utf-8','ignore'))
        #print('data is '+ list_data[2])
        count_pos += 1
        #print('pos is ' + str(count_pos))
        y.append(int(list_data[2].strip()))
    data_object = dataObjectContainer(X = X,y=y,user_names = user_name_list)
    return data_object
def get_list_of_data(dictOfFeatures):
    X_update = []
    y_update = []
    stringDoc = 'doc'
    i = 0
    hash_tag_list = []
    latest_tweets = []
    dataObject = dataObjectContainer()
    
    for values in xrange(0,len(dictOfFeatures)):
        prepare_doc = ''
        if(dictOfFeatures[stringDoc+str(values)].get(2,None) == None):
            hash_tag_list.append(0)
	if(dictOfFeatures[stringDoc+str(values)].get(3,None) == None):
	    latest_tweets.append('0')
        for keyVal in dictOfFeatures[stringDoc+str(values)].keys():
            setOb = dictOfFeatures[stringDoc+str(values)].get(keyVal,None)
            #if(setOb == None and (keyVal > 1):
            #   hash_tag_list.append(0)
            if(keyVal == 2):
                hash_tag_list.append(len(setOb))
	    if(keyVal == 3):
	    	tempstring = ''
		for elem in setOb:
			tempstring = elem +' '
		latest_tweets.append(tempstring)
            #else:
            #   pass
            if(setOb != None): # and keyVal != 1): and keyVal != 2 and keyVal != 3):
		# strings = ''
                for elem in setOb:
                    prepare_doc = prepare_doc + ' ' + elem.strip()
        X_update.append(prepare_doc.strip().decode('utf-8','ignore'))
    dataObject.hash_tag = hash_tag_list
    dataObject.X = X_update
    dataObject.latest_tweets = latest_tweets
    #print('hash_tag list: ' + str(len(hash_tag_list)))
    return dataObject
def parse_topic_features(path_name,size):
	dict_for_topic_features = {}
	file_for_topics = open(path_name)
	try:
		data = file_for_topics.readlines()
	finally:
		file_for_topics.close()
	matrix = np.zeros(shape=(len(data),size),dtype=float,order='F')
	for datum in data:
		data_list_items = datum.strip().split('\t')
		head,tail = os.path.split(data_list_items[1])
	        file_index = re.findall('[\d]+',tail)
		for elem in xrange(2,len(data_list_items),2):
			#print('element is :' + str(elem))
			#print('data we have now :' + data_list_items[elem+1])
			matrix[int(file_index[0])][int(data_list_items[elem])] = float(data_list_items[elem + 1])
			#print(matrix[int(file_index[0])])
		

	return matrix
def character_ngrams(data_source,version):

	vectorizer = CountVectorizer(analyzer='char',min_df=2,stop_words='english',ngram_range=(5,5))
	matrix = vectorizer.fit_transform(data_source)
	storeModels(clf = vectorizer,i_i = version,t_t = 'v')

	return matrix
polarity = {'positive','negative','neutral'}
def dict_parser(data_source):
	for range_elem in xrange(0,len(data_source)):
		data_source[range_elem] = data_source[range_elem].replace('"','')
		data_source[range_elem] = data_source[range_elem].replace('.',' ')
		#data_source[range_elem] = data_source[range_elem].replace('\'','')
	dict_vec = DictVectorizer()
	mat = dict_vec.fit_transform(Counter(s.split()) for s in data_source)
	arr_dict = dict_vec.inverse_transform(mat)# check for redundancy operation here
	    
	return arr_dict
def emoticon_count(path_name='Binary_classifier_for_health/DataSet_2_supriya.txt',emoticon_file = 'Binary_classifier_for_health/EmoticonLookupTable.txt',\
			                       emoticon_granual = 'Binary_classifier_for_health/emoticonsWithPolarity.txt'):
	global polarity
	pat1 = '\(\^'
	pat2 = ' \^\)'
	file_to_read = open(path_name)
	vectorizer = CountVectorizer(min_df=1,ngram_range=(1,2))
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
			#print('pattern is: ' + pat1 + pat2)
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
	# matrix = vectorizer.fit_transform(X)
	#reverse_arr = vectorizer.inverse_transform(matrix)
	for datum in emoticon_data:
		sub_datum = datum.split('\t')
		if(emoticon_vec.get(sub_datum[0],None)== None):
			emoticon_vec[sub_datum[0]] = sub_datum[1]
	user = {}
	#print('lenght of X is: ' + str(len(X)))																			
	#print('lenght of arr_dict is: ' + str(len(arr_dict)))
        h_arr = np.zeros((len(arr_dict),3))
	pos,neg,neu = 0,0,0
	counter_map = dict(emoticon_vec.items() + emoticon_granual_vec.items())
	for i in xrange(0,len(arr_dict)):
		for keys in arr_dict[i]:
			if(counter_map.get(keys,None) != None):#emoticon_vec.get(keys,None) != None or emoticon_granual_vec.get(keys,None) != None):
				#print('reverse doc pos and element:'+ str(i) +' '+ keys)
				if(counter_map[keys].lower().find('positive') != -1): #or counter_map[keys].lower().find('negative') != -1 or counter_map[keys].lower().find('neutral') != -1):
					h_arr[i][0] += 1
					#print('indexes :' + str(i) +' '+ str(0))
					pos += 1
				elif(counter_map[keys].lower().find('negative') != -1):
					#print('indexes :' + str(i) +' '+ str(1))
					h_arr[i][1] += 1
					neg += 1
				else:	
					# print('indexes :' + str(i) +' '+ str(2))
					h_arr[i][2] += 1					
					neu += 1
	print("number of elements have the emoticons: " + str(pos) +' ' + str(neg) +' '+str(neu))
       	return h_arr
def get_punctuation_feature(data_source):
	punctuation_feature = zeros((len(data_source),1))
	for each in xrange(0,len(data_source)):
		temp_data = data_source[each].split(' ')
		for sub_each in temp_data:
			if(len(sub_each) == 0): continue
			if(sub_each[len(sub_each)-1] == '!' or sub_each[len(sub_each)-1] == '?'):
				punctuation_feature[each] = 1
				break
	return punctuation_feature
def get_username_feature(data_source):
	arr_username = zeros((len(data_source),1))
	for each in xrange(0,len(data_source)):
		temp_data = data_source[each].split(' ')
		if(temp_data[0].find('@') != -1):
			arr_username[each] = 1
	return arr_username
def get_rt_feature(data_source):
	print(type(data_source))
	rt_feature = zeros((len(data_source),1))
	for each in xrange(0,len(data_source)):
		datum = data_source[each].split(' ')
		for sub_each in datum:
			if(sub_each.lower() == 'rt'):
				rt_feature[each] = 1
				break
	return rt_feature
def get_hash_feature(data_source):
	hash_feature = zeros((len(data_source),1))
	for each in xrange(0,len(data_source)):
		datum = data_source[each].split(' ')
		for sub_each in datum:
			if(len(sub_each)== 0): continue
			hash_feature[each] = 1 if sub_each[0] == '#' else 0
	return hash_feature
def get_usermention_feature(data_source):
	usermention_feature = zeros((len(data_source),1))
	for each in xrange(0,len(data_source)):
		datum = data_source[each].split(' ')
		for sub_each in datum:
			if(len(sub_each) == 0): continue
			usermention_feature[each] = 1 if sub_each[0] == '@' else 0
	return usermention_feature
def gene_feature_using_twitter_tokenizer(data,version):

	vectorizer = TwitterCountVectorizer(min_df = 2, stop_words='english',ngram_range=(1,3))
	temp_mat = vectorizer.fit_transform(data)
  	feature_names = temp_mat.get_feaure_names()

	return
def read_and_build_dict():
	dict_for_tags = {}
 	file_to_read = open('Binary_classifier_for_health/health.txt')
	try:
		data = file_to_read.readlines()
	finally:
		file_to_read.close()
	index = 0
	for each in data:
		temp_string = each.split('\t')
		if(len(temp_string) >= 2):
			if(dict_for_tags.get(temp_string[1],None) == None):
				dict_for_tags[temp_string[1]] =index
				index += 1
	file_to_write  = open('Binary_classifier_for_health/pos_tag_index.txt','w+')
	for (key,value) in dict_for_tags.items():
		file_to_write.write(key +'\t'+str(value)+'\n')
	file_to_write.close()
	return
def work_with_pos(data_list = None,file_path = None):

	data_source = read_data_source('Binary_classifier_for_health/DataSet_2_supriya.txt')
	tuples = parse_data(data_source)
        file_to_write = open('Binary_classifier_for_health/health_dataset_pos.txt','w+')
	index = 0
	dict_of_tags = {}
	'''for each in tuples.X:
		file_to_write = open('Binary_classifier_for_health/dir_health/health_dataset_pos'+str(index)+'.txt','w+')
		file_to_write.write(each)
		file_to_write.close()
		index += 1'''
	list_of_data = []
	for each in xrange(0,len(tuples.X)):
		argsList = ['Binary_classifier_for_health/ark-tweet-nlp-0.3.2/runTagger.sh',"--output-format",'conll', 'Binary_classifier_for_health/dir_health/health_dataset_pos'+str(each)+'.txt']
		temp_data = subprocess.Popen(argsList,stdout=subprocess.PIPE).communicate()
		#print argsList
		#print temp_data
		temp_data = temp_data[0]
		#print(temp_data)
		list_of_data.append(temp_data.split('\n'))
		#print('size of list '+str(len(temp_data.split('\n'))))
	'''for each in list_of_data:
		for sub_each in each:
			print(sub_each)'''
	get_data_set = []
	index = 0
	for each in list_of_data:
	 	temp_string = ''
		for sub_each in each:
			if(len(sub_each.split('\t')) >= 2):
				temp_list_data = sub_each.split('\t')
				if(dict_of_tags.get(temp_list_data[1],None)==None):
					dict_of_tags[temp_list_data[1]] = index 
					index += 1
				word_pos = temp_list_data[0] +'_'+ temp_list_data[1]
				temp_string = temp_string +' '+word_pos +' '
		file_to_write.write(temp_string+'\n')
		get_data_set.append(temp_string.strip())
	'''file_for_pos_tags = open('Binary_classifier_for_health/dir_health/pos_tag.txt','w+')
	for (key,val) in dict_of_tags.items():
		file_for_pos_tags.write(key+'\t'+str(val)+'\n')
	file_for_pos_tags.close()
	print(str(index)+'\n')'''
	file_to_write.close()
	
	'''for each in xrange(0,index):#len(tuples.X))
		temp_data = os.system('Binary_classifier_for_health/ark-tweet-nlp-0.3.2/./runTagger.sh --output-format conll Binary_classifier_for_health/dir_health/health_dataset_pos'+ str(each)+'.txt')		  
 		print(type(temp_data))'''
	dict_pos = {}
	return (get_data_set,dict_of_tags)
def get_pos_tags():
	#data_set , dict_for_tags = work_with_pos()
	read_pos_tags = open('Binary_classifier_for_health/pos_tag_index.txt')
	read_data = open('Binary_classifier_for_health/health_dataset_pos.txt')
	try:
		data_pos_tags = read_pos_tags.readlines()
	finally:
		read_pos_tags.close()
	try:
		data_set = read_data.readlines()
	finally:
		read_data.close()
        dict_for_tags = {}
	for each in data_pos_tags:
		temp_str = each.split('\t')
		dict_for_tags[temp_str[0]] = int(temp_str[1].strip())
        pos_tag_mat = np.zeros((len(data_set),len(data_pos_tags)))
	#print(str(pos_tag_mat.shape))
	for index , each in enumerate(data_set):
		temp_each = each.split('  ')
		for sub_each in temp_each:
			temp_str = sub_each[sub_each.rfind('_') + 1 :].rstrip()
			#print(temp_str)
			if(dict_for_tags.get(temp_str,None) != None):
				#print(str(dict_for_tags.get(temp_str,None)))
				pos_tag_mat[index][dict_for_tags[temp_str]] += 1
	#print(pos_tag_mat)
	return pos_tag_mat
def get_consecutive_pos_tag():
	
	read_pos_tags = open('Binary_classifier_for_health/pos_tag_index.txt')
	read_data = open('Binary_classifier_for_health/health_dataset_pos.txt')
	consecutive_pos_tag_elem = []
	try:
		data_pos_tags = read_pos_tags.readlines()
	finally:
		read_pos_tags.close()
	try:
		data_set = read_data.readlines()
	finally:
		read_data.close()
        dict_for_tags = {}
	for each in data_pos_tags:
		temp_str = each.split('\t')
		dict_for_tags[temp_str[0]] = int(temp_str[1].strip())
        pos_tag_mat = np.zeros((len(data_set),len(data_pos_tags)))
	#print(str(pos_tag_mat.shape))
	data_set_len = len(data_set)
	for index , each in enumerate(data_set):
		temp_each = each.split('  ')
		temp_elem = ''
		#print(temp_each)
		for i in xrange(0,len(temp_each)-2):
			temp_str_1 = temp_each[i][temp_each[i].rfind('_') + 1 :]
			for j in xrange(i + 1,i + 3):
				temp_str_1 += temp_each[j][temp_each[j].rfind('_') + 1 :]
				temp_elem = temp_elem + temp_str_1 +' '
		#print(temp_elem)
		consecutive_pos_tag_elem.append(temp_elem)

	return consecutive_pos_tag_elem
def tokens_with_tags():
	read_pos_tags  = open('Binary_classifier_for_health/pos_tag_index.txt')
	read_data = open('Binary_classifier_for_health/health_dataset_pos.txt')
	try:
		data_pos_tags = read_pos_tags.readlines()
	finally:
		read_pos_tags.close()
	try:
		data_set = read_data.readlines()
	finally:
		read_data.close()
	dict_for_tags = {}
	for each in data_pos_tags:
		temp_str = each.split('\t')
		dict_for_tags[temp_str[0]] = temp_str[1]
	pos_tag_mat = np.zeros((len(data_set),len(data_pos_tags)))
	#print(str(pos_tag_mat.shape))
	vectorizer = CountVectorizer(min_df= 1,stop_words='english',ngram_range=(1,2))
	mat = vectorizer.fit_transform(data_pos_tags)
	return mat
def pre_process_of_characters(data):
	
	for index_data, datum in enumerate(data):
		sub_datum = datum.split(' ')
		for index_datum,sub_sub_datum in enumerate(sub_datum):
			for i in xrange(1,len(sub_sub_datum) - 1):
				if(len(find_urls('don_know',sub_sub_datum[i])) != 0 ): continue
				if(sub_sub_datum[i-1] == sub_sub_datum[i] and sub_sub_datum[i] == sub_sub_datum[i+1]):
					sub_sub_datum.replace(sub_sub_datum[i],'')
			sub_datum[index_datum] = sub_sub_datum
		data[index_data] = ' '.join(sub_datum)
		#print(data[index_data])
	return
def positiveMeasurement(listq,indicator,score):
	for elem in xrange(0,len(listq.expecresultList)):
		if(listq.expecresultList[elem] == listq.preresultList[elem]):
			if(listq.expecresultList[elem] == 1):
				score.Tp += 1      
			else:
			 	score.Tn += 1
		else:
		  	if(listq.expecresultList[elem] == 1):
				score.Fn += 1
			else:
			 	score.Fp += 1
	return score
def fscore_measurement(p,r,beta):
    return ((1.0+math.pow(beta,2))*p*r)/((math.pow(beta,2)*p)+r)
def scoreMeasurement(measurementOfscore,beta_val):
    fscoreMeasurement = fScore()
    #fscoreMeasurement = positiveMeasurement(measurementOfscore,1,fscoreMeasurement)
    fscoreMeasurement = positiveMeasurement(measurementOfscore,0,fscoreMeasurement)
    f_score = precisionAndrecall(fscoreMeasurement,beta_val)
    return f_score.f1,f_score.p,f_score.r
def precisionAndrecall(fscoreMeasurement,beta_val):
    #print('Tp: %3.5f'%(fscoreMeasurement.Tp))
    #print('Tn: %3.5f'% (fscoreMeasurement.Tn))
    #print('Fp: %3.5f'%(fscoreMeasurement.Fp))
    #print('Fn: %3.5f '%(fscoreMeasurement.Fn))
    fscoreMeasurement.p = float(fscoreMeasurement.Tp)/(float(fscoreMeasurement.Tp+fscoreMeasurement.Fp))
    fscoreMeasurement.r = float(fscoreMeasurement.Tp)/(float(fscoreMeasurement.Tp+fscoreMeasurement.Fn))
    #print('Precision: %3.2f '% (fscoreMeasurement.p))
    #print('Recall: %3.2f' % (fscoreMeasurement.r))
    fscoreMeasurement.f1 = fscore_measurement(fscoreMeasurement.p,fscoreMeasurement.r,beta_val)
    #print('F-score: %3.5f' %(fscore_measurement(fscoreMeasurement.p,fscoreMeasurement.r)))
    return fscoreMeasurement
def get_efficient_combinations(strVal,strLen):
	list_of_combinations = []
	if(len(strVal)%2 != 0 ):
		str_op = strVal[1:]
	else:
	 	str_op = strVal[:]
	final_list_elem = []
	if(len(str_op)% 2 != 0):
		print('missing option value enter the option value and retry\n')
		sys.exit(0)
	#print('string lenght:' + str(str_op))
        for i in xrange(0,len(str_op),2):
		if(int(str_op[i+1].strip()) == 1):
			final_list_elem.append(str_op[i])
	print('final list: ' + str(final_list_elem))
	if(len(str_op) == 0 ):
		list_of_combinations.append([])
	for i in xrange(1,strLen + 1):
		temp_list = list(itertools.combinations(final_list_elem,i))
		list_of_combinations.extend(temp_list)
	return list_of_combinations, final_list_elem
def validation_check(list_of,global_feature_dict):
	for key,value in global_feature_dict.iteritems():
		global_feature_dict[key] = 1
	for each in list_of:
		if(global_feature_dict.get(each,None) == None):
			return True
	return False
def tweet_lenght(data_source):
	mat_return = zeros((len(data_source),1))
	for idx,each in enumerate(data_source):
		temp_tweets = each.split(' ')
		if(len(temp_tweets) > 10):
			mat_return[idx][0] = 1
		else:
		 	mat_return[idx][0] = 0
	return mat_return
def grab_result(result_set_tup):
	result_set = result_set_tup[0]
	shuffle_ind = result_set_tup[1]
        l = result_set_tup[2]
	file_i = result_set_tup[3]
	print('F1 score is : ' + str(result_set.F1)+"\n")
	print('P score is : ' + str(result_set.P) +'\n')
	l.acquire()
	write_result = open('Binary_classifier_for_health/avg_shuffle_output/4_result_sets'+str(file_i)+'.txt','a')
	#write_result.write('shuffling iterations :' + str(shuffle_ind)+'\n' )
	write_result.write('best_combinations: ')
	#result_set_list = result_set.best_combination.split(' ')
	p = re.compile('[a-zA-Z]+')
	for each in xrange(0,len(result_set.best_combination)):
		temp_ob = p.match(result_set.best_combination[each])
		if(temp_ob == None):
			continue
		if(each != len(result_set.best_combination) - 1):
		    	write_result.write(result_set.best_combination[each]+',')
		else:
			write_result.write(result_set.best_combination[each] +' ')
    	write_result.write('current_beta_score: ' + str(fabs(result_set.beta-1.0))+' ')
    	write_result.write('best_F-beta: ' + str(result_set.F1)+' ')
    	write_result.write('best_precision: ' + str(result_set.P)+' ')
    	write_result.write('best_recall: ' + str(result_set.R)+'\n')
	result_set.F1 = 0
	write_result.close()
	l.release()
	return
def work_with_combinations(list_of_combinations,y,sparse_mat,global_data_mat,beta_val,shuffle_index,l,buildingClassifier,kf,output_file_index):
	result_set = result_collector()
	print('execute now :')
	for each in list_of_combinations:
    		temp_sparse_mat = sps.csr_matrix(sparse_mat,copy=True)
    		for sub_each in each:
			print('each element :' + str(sub_each))
			print(str(global_data_mat[sub_each].shape))
    			temp_sparse_mat = hstack([temp_sparse_mat,global_data_mat[sub_each]])
		print('start of this training and testing')
		list_of_result = buildingClassifier(temp_sparse_mat,y,1.5,kf)
		print('i am breaking down here i guess')
    		fscore_result_F1 = []
		fscore_result_P = []
		fscore_result_R = []
	        #print(str(list_of_result))
    		for sub_sub_each in list_of_result:
			F1,P,R = scoreMeasurement(sub_sub_each,beta_val)
    			fscore_result_F1.append(F1)
			fscore_result_P.append(P)
			fscore_result_R.append(R)
		p,r,f1 = 0,0,0
		if(len(fscore_result_F1) == 0 ):
			print('continue')
			continue
		if(result_set.F1 < float(sum(fscore_result_F1))/len(fscore_result_F1)):
			result_set.beta = beta_val
			result_set.best_combination = each[:]
			print(str(result_set.best_combination))
			result_set.F1 = float(sum(fscore_result_F1))/len(fscore_result_F1)
			result_set.P = float(sum(fscore_result_P))/len(fscore_result_P)
        		result_set.R =  float(sum(fscore_result_R))/len(fscore_result_R)
		print('average f1 score ' + str(float(sum(fscore_result_F1))/len(fscore_result_F1)))
        	print('average P score ' + str(float(sum(fscore_result_P))/len(fscore_result_P)))
       		print('average R score ' + str(float(sum(fscore_result_R))/len(fscore_result_R)))
		del temp_sparse_mat
	
	return (result_set ,shuffle_index,l,output_file_index)

def main_starter(list_of_combinations =[('em','pos','rt','tw_len','punc')] ,output_file_index = 15):
  	
    '''if(string_command != None):
	    arg_val = string_command.strip().split(' ')
    else:
	    arg_val = sys.argv'''

    global_feature_dict = {'ch':0,'td':0,'pos':0,'em':0,'usern':0,'urls':0,'con_pos':0,'rt':0,'punc':0,'tw_len':0,'user_men':0,'hash_tag':0}
    global_data_mat = {}
    ''' arg_counter = len(arg_val)
    print(str(arg_val)) 
    list_of_combinations,final_list = get_efficient_combinations(arg_val,arg_counter)
    print('list of combinations: ')
    #print(str(arg_val[1:]))
    if(validation_check(final_list,global_feature_dict)):
	print('some invalid parameter value please enter the correct parameter values')
	sys.exit(0)'''
    print('Inception Point')
    result_set = result_collector()
    #work_with_pos()
    consecutive_pos_tag = get_consecutive_pos_tag()
    #file_to_write_for_noah = open('Binary_classifier_for_health/whole_data.txt','w+')
    data = read_data_source("Binary_classifier_for_health/DataSet_2_supriya.txt")
    print('Loading the urls')
    startLoadingUrls(1)
    print('lenght of data: ' + str(len(data)))
    tuples = parse_data(data) # replace the short urls with expanded urls
    print('Dictionary Features decomposition')
    pre_process_of_characters(tuples.X)
    dictOfFeature,list_of_tweets = dict_Of_Features(tuples.X) # dict of features 
    con_pos_tag = generates_features(consecutive_pos_tag,1.9)
    global_data_mat['con_pos'] = con_pos_tag
    rt_feature = get_rt_feature(tuples.X)
    global_data_mat['rt'] = rt_feature
    user_men = get_usermention_feature(tuples.X)
    global_data_mat['user_men'] = user_men
    hash_tag = get_hash_feature(tuples.X)
    global_data_mat['hash_tag'] = hash_tag
    tweets_lenght = tweet_lenght(tuples.X)
    global_data_mat['tw_len'] = tweets_lenght
    punctuation_feature = get_punctuation_feature(tuples.X)
    global_data_mat['punc'] = punctuation_feature
    emo_arr = emoticon_count() # emoticon matrix
    global_data_mat['em'] = emo_arr
    arr_username = get_username_feature(tuples.X) # user name matrix
    global_data_mat['usern'] = arr_username
    get_list = get_list_of_data(dictOfFeature)
    hash_train = np.array(get_list.hash_tag).reshape(989,1) #np.array(get_list.hash_tag[:800]).reshape(800,1) #hash matrix
    global_data_mat['ht'] = hash_train
    pos_tag_mat = get_pos_tags() # pos tag matrix
    global_data_mat['pos'] = pos_tag_mat
    sparse_mat = generates_features(list_of_tweets,1.5)#generates_features(get_list.X, 1.5)
    topic_mat_features = parse_topic_features('Binary_classifier_for_health/topic_per_distribution_3.txt',20)
    topic_mat_features = sparse.csr_matrix(topic_mat_features) # topic feature matrix
    global_data_mat['td'] = topic_mat_features
    #print get_list.latest_tweets
    sparse_mat_urls = generates_features(get_list.latest_tweets,1.6)#generates_features(get_list.latest_tweets[:800],1.6) # url matrix
    print('get lenght: ' + str(sparse_mat.shape))
    print('len of combinations ' + str(list_of_combinations))
    global_data_mat['urls'] = sparse_mat_urls
    #print(str(hash_train.shape))
    sparse_mat = sparse_mat.astype('float')
    train_character_ngrams = character_ngrams(get_list.X,1.7) #character_ngrams(get_list.X[:800],1.7) # character matrix
    global_data_mat['ch'] = train_character_ngrams
    best_combination = []
    #write_result = open('Binary_classifier_for_health/result_sets.txt','w+')
    shuffle_index = 1
    pool =  Pool(processes = 1)
    manager = Manager()
    lock = manager.Lock()
    #kf = cross_validation.StratifiedKFold(tuples.y,1,indices = True)
    buildingClassifier(sparse_mat,tuples.y,1.1)
		    
    print('start working on cross validation\n')
    while(True):
    	kf = cross_validation.StratifiedKFold(tuples.y,1,indices = True)
	#write_result.write('shuffling iteration: ' + str(shuffle_index)+'\n')
    	for beta_val in np.arange(1.50,2.0,0.5):
		li_com = list_of_combinations[:]
		s_mat = sps.csr_matrix(sparse_mat,copy=True)
		g_mat = global_data_mat.copy()
		b_val= beta_val
		s_index = shuffle_index
		tup_y = tuples.y[:]
		#print(li_com)
		#print('starting the pool of threads')
#	pool.apply_async(work_with_combinations,(li_com,tup_y,s_mat,g_mat,b_val,s_index,lock,buildingClassifier,kf,output_file_index),callback = grab_result ) # main worker process
		#print('enf of the pool of threads')
		#work_with_combinations(list_of_combinations,sparse_mat,global_data_mat,beta_val,lock)
	 	for each in list_of_combinations:
    			temp_sparse_mat = sps.csr_matrix(sparse_mat,copy=True)
    			for sub_each in each:
				print(sub_each)
				print(str(global_data_mat[sub_each].shape))
    				temp_sparse_mat = hstack([temp_sparse_mat,global_data_mat[sub_each]])
			list_of_result = buildingClassifier(temp_sparse_mat,tuples.y,1.5,kf)
    			fscore_result_F1 = []
			fscore_result_P = []
			fscore_result_R = []
		#print(str(list_of_result))
    			for sub_sub_each in list_of_result:
				F1,P,R = scoreMeasurement(sub_sub_each,beta_val)
    				fscore_result_F1.append(F1)
				fscore_result_P.append(P)
				fscore_result_R.append(R)
			p,r,f1 = 0,0,0
			if(result_set.F1 < float(sum(fscore_result_F1))/len(fscore_result_F1)):
				beta = beta_val
				best_combination = each[:]
				result_set.F1 = float(sum(fscore_result_F1))/len(fscore_result_F1)
				result_set.P = float(sum(fscore_result_P))/len(fscore_result_P)
        			result_set.R =  float(sum(fscore_result_R))/len(fscore_result_R)
			print('average f1 score ' + str(float(sum(fscore_result_F1))/len(fscore_result_F1)))
        		print('average P score ' + str(float(sum(fscore_result_P))/len(fscore_result_P)))
       			print('average R score ' + str(float(sum(fscore_result_R))/len(fscore_result_R)))
		#test_phase(temp_sparse_mat[800:0],tuples.y[800:],1.5)
      			del temp_sparse_mat	
    		write_result.write('best combinations : ')
    		write_result.write(str(best_combination)+'\n')
    		write_result.write('best F-beta score ' + str(beta_val)+'\n')
    		write_result.write('best F-beta: ' + str(result_set.F1)+'\n')
    		write_result.write('best precision : ' + str(result_set.P)+'\n')
    		write_result.write('best recall : ' + str(result_set.R)+'\n')
		result_set.F1 = 0
	if(shuffle_index == 1000):
		#print('i am done with ' +str(shuffle_index)+' different shuffling'+ '\n')
		break
	sparse_mat,tuples.y = shuffle(sparse_mat,tuples.y,random_state = 0)
	shuffle_index += 1
    pool.close()
    pool.join()
    print('done with shuffling operations thanks \n')
    print('i am done with  the processing \n')
    #write_result.close()
    return
    
#read_and_build_dict()
main_starter() # this is the main starting point for this programe

#work_with_pos()
#get_pos_tags()
#parse_topic_features('Binary_classifier_for_health/doc_per_topic_distribution.txt')
#print('end of execution of running programe')
