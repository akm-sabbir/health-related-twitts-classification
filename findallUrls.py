#! /usr/bin/env python
# -*- coding: utf-8 -*-
import nltk
import re
def detect_urls(tweets,user_names):
    url_list = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweets)
    return (user_names,url_list)

def get_urls():
    file_to_read = open('C:\\Binary_classifier_for_health\\DataSet_2_supriya.txt','r')
   # file_to_read_empty = open('C:\\TwitterBigData\\user_name_data_profiles\\user_tweets_empty_details.txt','r')
    file_to_write_urls = open('C:\\Binary_classifier_for_health\\urls_only.txt','w+')
    try:
        data = file_to_read.readlines()
    finally:
        file_to_read.close()
    '''try:
        data.extend(file_to_read_empty.readlines())
    finally:
        file_to_read_empty.close()'''
    line = 0
    for datum in data:
        temp_list = datum.split("\t")
        print(str(line))
        tuple_ob = detect_urls(temp_list[3], temp_list[1])
        #file_to_write_urls.write(tuple_ob[0]+'\t')
        for each in xrange(0,len(tuple_ob[1])):
            file_to_write_urls.write(tuple_ob[0]+'\t'+tuple_ob[1][each]+'\n')
            ''' if(each != len(tuple_ob[1]) - 1):
                file_to_write_urls.write(tuple_ob[1][each]+' ')
            else:
                file_to_write_urls.write(tuple_ob[1][each])'''
        line +=1
        #file_to_write_urls.write('\n')
    file_to_write_urls.close()
    return

get_urls()