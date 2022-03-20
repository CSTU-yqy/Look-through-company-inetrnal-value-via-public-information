import pandas as pd
import numpy as np
import time
import os
import sys
sys.path.append("F:\paper\code\getCnki\handle_data")
from tool.BAGGING import BAGGING
import jieba
jieba.load_userdict("F:\paper\data\important\cover_name.txt")
import multiprocessing as mlp
import threading
from tool.toolBox import DATA
from tool.get_different_mindf import GetData
DATA = DATA()
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn import metrics
import warnings
import pickle
from scipy.sparse import csr_matrix
import copy
from bidict import bidict
warnings.filterwarnings("ignore")

class Model():

    def __init__(self,org,line_standard = 0.1,min_df = 3):
        self.author = "Qianyu Yang"
        self.org = org.dropna(axis = 0,subset = ["id"])
        self.line_standard = line_standard
        gd = GetData(min_df = min_df,cut_method="global",all_data_address = "F:\paper\code\getCnki\handle_data\single_file\clean_data_2\data\\all_data_cut_global.pkl")



        self.all_data = gd.data
    #all_data = pd.read_pickle("/Users/qianyuyang/Desktop/paper/data/important/sentence_target_cut_already.pkl")
        self.all_data = self.all_data.drop_duplicates(["sentence"])
        self.all_data["id"] = self.all_data.index
        ##################################################################################################################
        #这里加工整体的tf-idf矩阵
        
        self.all_tf = TfidfVectorizer()
        #############################################
        self.all_tf_parse_matrix = self.all_tf.fit_transform(self.all_data.text.tolist())
        #######可以省略这一步体现系数矩阵的优越性##############
        self.word_news_parse_matrix = csr_matrix((np.array([1] * len(self.all_tf_parse_matrix.data)),self.all_tf_parse_matrix.indices,self.all_tf_parse_matrix.indptr),shape = self.all_tf_parse_matrix.shape)
        self.all_word_list = self.all_tf.get_feature_names()
        
        c = copy.deepcopy(self.all_data)
        c["map_index"] = list(range(len(c)))
        self.mp = bidict(c["map_index"].to_dict())
        #self.rmp = dict(zip(mp.values(),mp.keys()))

    @property
    def result(self):
        ad = copy.deepcopy(self.all_data)
        word_vec = pd.DataFrame(columns = ["word","score"])
        word_vec.word = self.all_word_list
        #word_vec.word = tf.get_feature_names()
        #############可以优化a.sum(axis = 0)*label###################
        word_vec.score = self.all_tf_parse_matrix[[self.mp[i] for i in self.org.id]].T.dot(np.array(self.org.label.tolist())) / np.array(self.word_news_parse_matrix[[self.mp[i] for i in self.org.id]].sum(axis = 0))[0]
        #word_vec.score = np.array(org.label) * np.array(all_tf_parse_matrix[[mp[i] for i in org.id]].sum(axis = 0))[0] /np.array(word_news_parse_matrix[[mp[i] for i in org.id]].sum(axis = 0))[0]
        ############################################################
        word_vec = word_vec.sort_values("score")
        
        cover_news = pd.DataFrame(columns = ["id","round"])
        cover_news["id"] = self.org.id
        cover_news["round"] = 0
        round_count = 0
        while True:
            try:
                round_count += 1
                line = int(len(self.org[self.org.label == -1]) * self.line_standard)
                search_id = word_vec[word_vec.score < 0].index.tolist()
                rest_data_id = list(set(self.mp.values()) - set([self.mp[i] for i in self.org.id]))
                rest_word_all_news_parse_matirx = self.word_news_parse_matrix[:,search_id]
                sa = np.array(rest_word_all_news_parse_matirx.sum(axis = 1).T)[0]
                increase_stat = dict(zip([self.mp.inverse[i] for i in rest_data_id],sa[rest_data_id]))
                increase_stat = {key:value for key,value in increase_stat.items() if value != 0}
                new_cover_news = pd.DataFrame(columns = ["id","round"])
                new_cover_news["id"] = sorted(increase_stat,key=increase_stat.__getitem__,reverse=True)[:line]
                new_cover_news["round"] = round_count
                
                cover_news = pd.concat([cover_news,new_cover_news])
                new_round_data = ad[ad.id.isin(cover_news[cover_news["round"] == round_count].id.tolist())]
                labels = self.org.label

                """
                预测是用局部矩阵进行预测的
                """
                tf = TfidfVectorizer()
                train_features = tf.fit_transform(self.org.text.tolist())
                model = MultinomialNB(alpha = 0.001).fit(train_features, labels)
                test_features = tf.transform(new_round_data.text.tolist())
                new_round_data["label"] = model.predict(test_features)
                new_labels = new_round_data.label
                self.org = pd.concat([new_round_data,self.org])

                labels = self.org.label

                word_vec = pd.DataFrame(columns = ["word","score"])
                #word_vec.word = tf.get_feature_names()
                #word_vec.score = np.matmul(np.array(org.label).T,tf_matrix.toarray())
                word_vec.word = self.all_word_list
                #word_vec.word = tf.get_feature_names()
                ###################可以优化，用a.sum(axis = 0)*label########################
                #word_vec.score = np.matmul(np.array(org.label).T,all_tf_matrix[[mp[i] for i in org.id]])
                #word_vec.score = np.array(org.label) * np.array(all_tf_parse_matrix[[mp[i] for i in org.id]].sum(axis = 0))[0] /np.array(word_news_parse_matrix[[mp[i] for i in org.id]].sum(axis = 0))[0]
                word_vec.score = self.all_tf_parse_matrix[[self.mp[i] for i in self.org.id]].T.dot(np.array(self.org.label.tolist())) / np.array(self.word_news_parse_matrix[[self.mp[i] for i in self.org.id]].sum(axis = 0))[0]
                ##################################################################
                word_vec = word_vec.sort_values("score")
                
            except:
                
                break
        
        """
        注意顺序不要反
        """
        iteration_result = pd.merge(ad,self.org[["id","label"]],on = "id",how = "left").fillna(0)
        self.org = pd.merge(ad,self.org[["id","label"]],on = "id",how = "left").fillna(1)
        #word_vec.score = self.all_tf_parse_matrix[[self.mp[i] for i in self.org.id]].T.dot(np.array(self.org.label.tolist())) / np.array(self.word_news_parse_matrix[[self.mp[i] for i in self.org.id]].sum(axis = 0))[0]
        word_vec.score = word_vec.score.fillna(0)
        return {
                "prediction":self.org,
                "iteration_result":iteration_result,
                "word_vector":word_vec
                }

