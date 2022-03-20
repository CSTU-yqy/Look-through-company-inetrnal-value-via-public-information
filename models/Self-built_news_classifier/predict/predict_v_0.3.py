#这里尝试剔除只出现一次的一些词
import pandas as pd
import numpy as np
import time
import os
import sys
sys.path.append("F:\paper\code\getCnki\handle_data")
from tool.BAGGING import BAGGING
import jieba
jieba.load_userdict("F:\paper\data\important\cover_name.txt")
import multiprocessing as mp
import threading
from tool.toolBox import DATA,TOOl
from tool.model_kernel_complement import Model
from tool.get_different_mindf import GetData
from tool.BAGGING import kfold
DATA = DATA()
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn import metrics
import warnings
import pickle
from scipy.sparse import csr_matrix
import multiprocessing as mlp
import copy
from bidict import bidict
warnings.filterwarnings("ignore")
import joblib



#file_handle = open("exp.txt","a")
# sample_size = 10
# vote_num = 5
# process_num = sample_size
def trans_num(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
"""
org在本样本中一直代表初始的训练样本

"""
def predict(min_df,org,seed,vote_num = 7):
    
    gd = GetData(min_df = min_df,cut_method="global",all_data_address = "F:\paper\code\getCnki\handle_data\single_file\clean_data_2\data\\all_data_cut_global.pkl")
    #kf = kfold(org,n_split)
    #grand_org = kf.train_data[seed]
    #test_data = kf.test_data[seed]
    
    grand_org = copy.deepcopy(org)
    all_data = gd.data
#all_data = pd.read_pickle("/Users/qianyuyang/Desktop/paper/data/important/sentence_target_cut_already.pkl")
    all_data = all_data.drop_duplicates(["sentence"])
    all_data["id"] = all_data.index
    ##################################################################################################################
    #这里加工整体的tf-idf矩阵
    
    all_tf = TfidfVectorizer()
    #############################################
    all_tf_parse_matrix = all_tf.fit_transform(all_data.text.tolist())
    #######可以省略这一步体现系数矩阵的优越性##############
    word_news_parse_matrix = csr_matrix((np.array([1] * len(all_tf_parse_matrix.data)),all_tf_parse_matrix.indices,all_tf_parse_matrix.indptr),shape = all_tf_parse_matrix.shape)
    all_word_list = all_tf.get_feature_names()

    """
    这里先分出测试集和训练集test_org和grand_org
    """
    iteration_result_pool = []
    org_pool = []
    word_vector_sum = pd.DataFrame()
    for vote in range(vote_num):
        """
        这里再根据训练集划分出训练集和验证集train_org,test_org
        """
        test_org = grand_org.sample(int(0.2 * len(grand_org)),replace = False)
        train_org = grand_org[grand_org.id.isin(test_org.id.tolist()) == False]
        if not os.path.exists("min_df = %s/para_dict"%min_df):
            os.mkdir("min_df = %s/para_dict"%min_df)
        param_dict = dict()
        with open("min_df = %s/para_dict/para_dict_%s.pkl"%(min_df,seed),"wb") as f:
                pickle.dump(param_dict,f)
        
        for line_standard in range(20,0,-1):
            line_standard = round(line_standard * 0.1,1)
            
            print("_____________________________________\n")
            print("the line standard is %s\n"%line_standard)
            print("_____________________________________\n")
            
            """
            每一个seed的每一个vote都需要进行参数学习来获得最佳步长，这一步就是加载出目前的sample/vote中前面不同步长在验证集上的表现
            """
            with open("min_df = %s/para_dict/para_dict_%s.pkl"%(min_df,seed),"rb") as f:
                param_dict = pickle.load(f)
            ad = copy.deepcopy(all_data)
            org = copy.deepcopy(train_org)
            torg = copy.deepcopy(test_org)
            
            org = TOOl().clean(min_df = min_df,org = org)
            m = Model(org = org,line_standard = line_standard,min_df = min_df)
            k = m.result
            org = k["prediction"]
            word_vec = k["word_vector"]
            """
            这里会显示负面新闻的总数量
            """
            print("_____%s_________"%len(org[org.label == -1]))
            
            td = pd.merge(torg,org[["sentence","label"]],on = "sentence",how = "left")
            
            td = td.dropna(axis = 0,subset=["label_x","label_y"])
        
            f1s = f1_score(td.label_x,td.label_y,average = "macro")
            
            param_dict[line_standard] = f1s
            with open("min_df = %s/para_dict/para_dict_%s.pkl"%(min_df,seed),"wb") as f:
                pickle.dump(param_dict,f)
        stat = pickle.load(open("min_df = %s/para_dict/para_dict_%s.pkl"%(min_df,seed),"rb"))
        best_key = max(stat,key = stat.get)
        """
        在获得了best步长之后有一个优化的地方是将验证集扔回去用全部的grand_org进行训练
        """
        again_result = Model(org = TOOl().clean(min_df = min_df,org = grand_org),line_standard = best_key,min_df = min_df).result
        iteration_word_vector = again_result["word_vector"]
        org_pool.append(again_result["prediction"])
        iteration_result_pool.append(again_result["iteration_result"])

        if  len(word_vector_sum) == 0:

            word_vector_sum = iteration_word_vector.sort_values("word")
        else:
            word_vector_sum.score = word_vector_sum.score + iteration_word_vector.sort_values("word").score
    seed_org = copy.deepcopy(org_pool[0])
    seed_org["label"] = 0
    seed_iteration_result = copy.deepcopy(iteration_result_pool[0])
    seed_iteration_result["label"] = 0
    for i in range(vote_num):
        
        seed_org["label"] = np.array(seed_org["label"]) + np.array(org_pool[i]["label"])
        seed_iteration_result["label"] = np.array(seed_iteration_result["label"]) + np.array(iteration_result_pool[i]["label"])
    #写的方式


    seed_org["label"] = seed_org["label"].apply(trans_num)
    """
    2022年0223 13：29修改，解决pickle memory error
    """
    joblib.dump(seed_org,"predict_%s_%s.pkl"%(min_df,seed))
    #seed_org.to_pickle("predict_%s_%s.pkl"%(min_df,seed))
    joblib.dump(word_vector_sum,"word_vector_%s_%s.pkl"%(min_df,seed))
    #word_vector_sum.to_pickle("word_vector_%s_%s.pkl"%(min_df,seed))
    joblib.dump(seed_iteration_result,"iteration_result_%s_%s.pkl"%(min_df,seed))
    #seed_iteration_result.to_pickle("iteration_result_%s_%s.pkl"%(min_df,seed))
    unanimous_result = seed_iteration_result[(seed_iteration_result.label == vote_num) | (seed_iteration_result.label == (vote_num * -1))]
    unanimous_result["label"] = unanimous_result["label"].apply(trans_num)
    joblib.dump(unanimous_result,"unanimous_result_%s_%s.pkl"%(min_df,seed))
    #unanimous_result.to_pickle("unanimous_result_%s_%s.pkl"%(min_df,seed))
    if not os.path.exists("min_df = %s/para_dict"%min_df):
        os.mkdir("min_df = %s/para_dict"%min_df)
    #return seed_org,word_vector_sum
    return