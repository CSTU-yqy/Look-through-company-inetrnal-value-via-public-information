from dataclasses import replace
from random import sample
import argparse

parser = argparse.ArgumentParser(description='model train')
parser.add_argument('--model_alpha', type=float, required=True, help='choose a model_alpha')
args = parser.parse_args()

if __name__ == "__main__":
        
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
    from tool.predict_kernel_complement_globalTF import predict
    warnings.filterwarnings("ignore")
    import joblib
    #file_handle = open("exp.txt","a")
    sample_size = 3
    vote_num = 3
    process_num = 1
    iteration_standard = [-1]
    model_alpha = args.model_alpha

    
    
    

    """
    seed代表我要做多少组验证，vote代表在产生每一组预测的过程中我要进行多少次模型来投票产生最终结果
    """
    def trans_num(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0


    for min_df in [3]:
        #ud = BAGGING(sample_size = sample_size).data
        pool = mlp.Pool(process_num)
        #grand_org = ud["train_data"][seed]
        #test_data = ud["test_data"][seed]
        if not os.path.exists("min_df = %s"%min_df):
            os.mkdir("min_df = %s"%min_df)
        org = BAGGING(sample_size = sample_size,org_address = "F:\paper\code\getCnki\handle_data\single_file\\new_predict\judge good bad news bert\sample_data.pkl")._org
        """
        这里开始初始化
        """
        #test_data_set = []
        train_data_set = []
        try:
            print(1)
            #test_data_set = [pd.read_pickle("test_data_%s_%s.pkl"%(min_df,seed)) for seed in range(sample_size)]
            train_data_set = [pd.read_pickle("train_data_%s_%s.pkl"%(min_df,seed)) for seed in range(sample_size)]
            unanimous_result_set = [joblib.load("unanimous_result_%s_%s.pkl"%(min_df,seed)) for seed in range(sample_size)]
            unanimous_result_set = [i[i.label == -1] for i in unanimous_result_set]
            """
            在这里做合并是因为如果需要初始化的话这里肯动是没有unanimous_result的
            """
            train_data_set = [pd.concat([train_data_set[i],unanimous_result_set[i][unanimous_result_set[i].label.isin(iteration_standard)].sample(len(train_data_set[i][train_data_set[i].label == -1]))]).drop_duplicates("sentence") for i in range(sample_size)]
            for seed in range(sample_size):
                train_data_set[seed].to_pickle("train_data_%s_%s.pkl"%(min_df,seed))
        except:
            print(2)
            good_org = org[org.label == 1]
            bad_org = org[org.label == -1]
            for i in range(sample_size):
                """
                20220224 14：38修改
                
                """
                """
                20220224 15:16修改回原来的版本
                """
                #good_train = good_org.sample(int(len(good_org) * 0.8)).drop_duplicates()
                #bad_train = bad_org.sample(int(len(bad_org) * 0.8)).drop_duplicates()
                good_train = good_org.sample(len(good_org),replace = True).drop_duplicates()
                bad_train = bad_org.sample(len(bad_org),replace = True).drop_duplicates()
                good_test = good_org[good_org.id.isin(good_train.id.tolist()) == False]
                bad_test = bad_org[bad_org.id.isin(bad_train.id.tolist()) == False]
                pd.concat([good_train,bad_train]).to_pickle("train_data_%s_%s.pkl"%(min_df,i))
                pd.concat([good_test,bad_test]).to_pickle("test_data_%s_%s.pkl"%(min_df,i))
                train_data_set.append(pd.concat([good_train,bad_train]))
                #test_data_set.append(pd.concat([good_test,bad_test]))
        
        param_dict = dict()
        for i in range(process_num):
            param_dict[i] = [[train_data_set[k] for k in range(sample_size) if k % process_num == i],[kk for kk in range(sample_size) if kk % process_num == i]]
        p = [pool.apply_async(predict,args = (min_df,b[0],b[1],vote_num,model_alpha,)) for b in param_dict.values()]
        #p = [pool.apply_async(predict,args = (min_df,train_data_set[b],b,vote_num,)) for b in range(sample_size)]
        final_res = [sub_p.get() for sub_p in p]
        """
        20220223 12:13修改
        
        """
        word_vector_list = [i[1] for i in final_res]
        final_word_vector = pd.DataFrame(columns = ["word","score"])
        final_word_vector.word = word_vector_list[0].word
        
        final_word_vector.score = sum([i.score.apply(trans_num) for i in word_vector_list])
        final_word_vector.to_pickle("word_vector_final.pkl")
        
"""
20220224 15:26修改
"""
        