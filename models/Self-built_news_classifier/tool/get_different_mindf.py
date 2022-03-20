
#这是一个输入词出现频率输出清洗完的总体数据的一个包

import pandas as pd
import numpy as np
import time
import os
import sys
sys.path.append("F:\paper\code\getCnki\handle_data")
import jieba
jieba.load_userdict("F:\paper\data\important\cover_name.txt")
import multiprocessing as mp
import threading
from tool.toolBox import DATA
DATA = DATA()
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import warnings
import pickle
import copy
warnings.filterwarnings("ignore")
import joblib




class GetData():
    
    def __init__(self,all_data_address,min_df = 2,max_df = 0.2,cut_method = "precise"):
        self.author = "Qianyu Yang"
        self.stop_word_list = DATA.stop_word_list
        self.cover_name_list = DATA.cover_name_list
        self.min_df = min_df
        self.max_df = max_df
        self.want_word = list()
        self.not_want_word = list()
        self.cut_method = cut_method
        #self.all_data = pd.read_pickle("F:/paper/data/important/sentence_target_cut_already.pkl") if self.cut_method == "precise" else pd.read_pickle("F:/paper/data/important/sentence_target_cut_global_already.pkl")
        #self.all_data = pd.read_pickle(all_data_address)
        self.all_data = joblib.load(all_data_address)


    def cut_word(self,text,method = "precise"):
                
        word_list = list()

        if method == "precise":
            word_list = jieba.cut(text,cut_all=False)
            

        elif method == "global":
            word_list = jieba.cut(text,cut_all=True)

        elif method == "single":
            word_list = list(text)
        
        return " ".join([i for i in word_list if i not in self.stop_word_list and i not in self.cover_name_list and u'\u4e00' <= i <= u'\u9fff' and len(i) > 1])
    
    
    
    def get_word_list(self):
        #all_data = pd.read_pickle("/Users/qianyuyang/Desktop/paper/data/important/sentence_target_cut_already.pkl")
        all_data = copy.deepcopy(self.all_data)
        all_data = all_data.drop_duplicates(["sentence"])
        all_data["id"] = all_data.index
        #all_data.text = all_data.sentence.apply(lambda x:self.cut_word(x,method = self.cut_method))
        tf1 = TfidfVectorizer() # 使用默认参数
        tf_matrix1 = tf1.fit_transform(all_data.text.tolist())
        all_word = set(tf1.get_feature_names())
        
        tf2 = TfidfVectorizer(max_df=self.max_df,min_df=self.min_df) # 使用默认参数
        tf_matrix2 = tf2.fit_transform(all_data.text.tolist())
        want_word = set(tf2.get_feature_names())

        not_want_word = all_word - want_word
        
        self.want_word = want_word
        
        self.not_want_word = not_want_word
        
        return
        
    def clean(self,df):
        self.get_word_list()
        df.text = df.text.apply(lambda x: " ".join([i for i in x.split(" ") if i in self.want_word and len(i) > 1]))
        return df
    
    
    @property
    def data(self):
        all_data = copy.deepcopy(self.all_data)
        #all_data = pd.read_pickle("/Users/qianyuyang/Desktop/paper/data/important/sentence_target_cut_global_already.pkl")
        #all_data = pd.read_pickle("/Users/qianyuyang/Desktop/paper/data/important/sentence_target_cut_already.pkl")
        kk = self.clean(all_data)
        #kk.to_pickle("/Users/qianyuyang/Desktop/paper/data/important/sentence_target_cut_%s_already_mindf%s.pkl"%(self.cut_method,self.min_df))
        #kk.to_pickle("/Users/qianyuyang/Desktop/paper/data/important/sentence_target_cut_already_mindf%s.pkl"%self.min_df)
        return kk