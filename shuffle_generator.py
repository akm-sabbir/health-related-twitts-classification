#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
from nltk.tokenize import RegexpTokenizer
from nltk import bigrams,trigrams
import math
import numpy as np
import scipy as sp
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import pickle
import sklearn
from sklearn.cross_validation import StratifiedKFold
from binary_classifier_for_health_data_01 import main_starter

def generate_folds(data):
        i = 0
	while(i < 1000):
		kf = cross_validation.StratifiedKFold(len(data),5,indices = True)
		model_saver = open('Binary_classifier_for_health/kf_shuffles/folds_'+str(i),'wb')
		i += 1
		pickle.dump(kf,model_saver,pickle.HIGHEST_PROTOCOL)
	
	return

def calculate_percentage_features():
	with open('Binary_classifier_for_health/shuffle_output/all') as my_files:
		data = my_files.readlines()
		
	return data
def get_counts(list_of_features,global_feature_dict):
	for each in list_of_features:
		for sub_each in each:
			#print(sub_each)
			sub_each = sub_each[1:len(sub_each)-1]
			global_feature_dict[sub_each.strip()] += 1
	return global_feature_dict
def calculate_feature_percentage(feature_counts,data_size):
	print('data size: ' + str(data_size))
	percentage_list = [0.4,1,10,20,30,40,50,60,70,80,90,100]
	combination_list = [[] for each in range(1) for x in range(12)]
	for key,value in feature_counts.iteritems():
		feature_counts[key] = float((float(value)/data_size)*100)
		for each in xrange(0,len(percentage_list)-1):
			if(feature_counts[key] > percentage_list[each] ):#and feature_counts[key] < percentage_list[each + 1]):
				combination_list[each].append(key)
	string_dict = {}
	for each in xrange(0,len(percentage_list) - 1):
		string_dict[each] = ''
	for idx, each in enumerate(combination_list):
	   if(len(each) != 0):
		string_commands = []
		for sub_each in each:
			string_commands.append(sub_each) # = string_commands +sub_each + ' 1'+' '
		#print(string_commands)
		if(string_dict.get(idx,None) != None):
			string_dict[idx] = string_commands
	
	return string_dict
def main_operation():
	list_of_features = []
	global_feature_dict = {'ht':0,'ch':0,'td':0,'pos':0,'em':0,'usern':0,'urls':0,'con_pos':0,'rt':0,'punc':0,'tw_len':0}
	'''	with open('Binary_classifier_for_health/DataSet_2_supriya.txt')	as my_file:
		data = my_file.readlines()
		generate_folds(data)'''
	list_of_data = calculate_percentage_features()
	for idx,each in enumerate(list_of_data):
		sub_list = each.split(' ')
		string = sub_list[1].strip('(')
		string = string.strip(')')
		#print(string)
		list_of_features.append(string.split(','))
	feature_counts = get_counts(list_of_features,global_feature_dict)
	print(feature_counts)
	list_of_command = calculate_feature_percentage(feature_counts,len(list_of_data))
	list_of = []
	for key,values in list_of_command.iteritems():
		tup = tuple(values)
		list_of.append(tup)
		print(list_of)
		main_starter(list_of,key)
		list_of = []
	return
main_operation()
