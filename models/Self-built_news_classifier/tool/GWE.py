import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import time
import os
import sys
sys.path.append("/Users/qianyuyang/Desktop/paper/code/getCnki/handle_data")
from BAGGING import BAGGING
import jieba
jieba.load_userdict("/Users/qianyuyang/Desktop/paper/data/important/cover_name.txt")
import multiprocessing as mp
import threading
from toolBox import DATA
from get_different_mindf import GetData
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
import multiprocessing as mp
from sklearn.feature_extraction.text import CountVectorizer
class get_weighted_edge():
    def __init__(self,core):
        self.author = "qianyu"
        self.core = core
        di = dict()
        min_df = 5
        sample_size = 5
        gd = GetData(min_df = min_df,cut_method="global")
                
        want_word = gd.want_word



        all_data = gd.data
        #all_data = pd.read_pickle("/Users/qianyuyang/Desktop/paper/data/important/sentence_target_cut_already.pkl")
        all_data = all_data.drop_duplicates(["sentence"])
        all_data["id"] = all_data.index
        for seed in range(sample_size):
            stat = pickle.load(open("/Users/qianyuyang/Desktop/paper/code/getCnki/handle_data/test21/21.1/min_df = %s/para_dict/para_dict_%s.pkl"%(min_df,seed),"rb"))
            best_key = max(stat,key = stat.get)
            di[seed] = pd.read_csv("/Users/qianyuyang/Desktop/paper/code/getCnki/handle_data/test21/21.1/min_df = %s/%s_%s_wv.csv"%(min_df,seed,best_key))
        word = set(di[0][di[0].score < 0].word.tolist())
        ss = dict()
        for i in range(0,sample_size):
            data = di[i]
            sss = set(data[data.score<0].word.tolist())
            for j in sss:
                if j not in ss.keys():
                    ss[j] = 0
                #ss[j] += data.loc[data.word == j,"score"].iloc[0]
                ss[j] += 1
        ws = set(sorted(ss,key=ss.__getitem__,reverse=False))
        wl = [k for k,v in ss.items() if v >= 1]
        predict_data = pd.read_pickle("/Users/qianyuyang/Desktop/paper/code/getCnki/handle_data/test21/21.2/predict_5_3.pkl")
        bad_data = predict_data[predict_data.label == -1]
        #bad_data.text = bad_data.text.apply(lambda x:" ".join(list(set(x.split(" ")) & set(wl))))
        bad_data.text = bad_data.text.apply(lambda x:" ".join([i for i in x.split(" ") if i in set(wl)]))

        countvec = CountVectorizer()

        self.x = countvec.fit_transform(bad_data.text.tolist())
        
        
    
    @classmethod
    def find_set(self,x:set) -> set:
        if len(x) <= 1:
            return set()
        else:
            y = list(x)
            res = {(y[i],y[j]) for i in range(1,len(y)) for j in range(i)}
            return res
        
    @property
    def final_al_ix(self):
        di = dict()
        min_df = 5
        sample_size = 5
        gd = GetData(min_df = min_df,cut_method="global")
                
        want_word = gd.want_word



        all_data = gd.data
        #all_data = pd.read_pickle("/Users/qianyuyang/Desktop/paper/data/important/sentence_target_cut_already.pkl")
        all_data = all_data.drop_duplicates(["sentence"])
        all_data["id"] = all_data.index
        for seed in range(sample_size):
            stat = pickle.load(open("/Users/qianyuyang/Desktop/paper/code/getCnki/handle_data/test21/21.1/min_df = %s/para_dict/para_dict_%s.pkl"%(min_df,seed),"rb"))
            best_key = max(stat,key = stat.get)
            di[seed] = pd.read_csv("/Users/qianyuyang/Desktop/paper/code/getCnki/handle_data/test21/21.1/min_df = %s/%s_%s_wv.csv"%(min_df,seed,best_key))
        word = set(di[0][di[0].score < 0].word.tolist())
        ss = dict()
        for i in range(0,sample_size):
            data = di[i]
            sss = set(data[data.score<0].word.tolist())
            for j in sss:
                if j not in ss.keys():
                    ss[j] = 0
                #ss[j] += data.loc[data.word == j,"score"].iloc[0]
                ss[j] += 1
        ws = set(sorted(ss,key=ss.__getitem__,reverse=False))
        wl = [k for k,v in ss.items() if v >= 1]
        
        x = copy.deepcopy(self.x)
        
        s = list(zip(x.nonzero()[1],x.nonzero()[0]))
        wt = list({c[0] for c in s})
        res = []
        for tar in wt:
            #if tar % 25 == 24:
                #print(tar)
            ee = {c[1] for c in s if c[0] == tar}
            res.append(self.find_set(ee))
        al_ix = []
        for i in range(len(res)):
            al_ix += list(res[i])
        final_al_ix = list(set(al_ix))
        
        return final_al_ix
    
    def get_cos_cor(self,lc):
        resu = []
        co = 0
        for k in lc:
            co += 1
            if co % 10000 == 9999:
                print(co // 10000)
            i = k[0]
            j = k[1]
            resu.append((i,j,self.x[i].dot(self.x[j].T).sum() / (np.sqrt(self.x[i].dot(self.x[i].T).sum()) * np.sqrt(self.x[j].dot(self.x[j].T).sum()))))
        #resu.to_pickle("ex.pkl")
        return resu
    
    def get(self,ai):
        #x = copy.deepcopy(self.x)
        #final_al_ix = self.final_al_ix
        
        
        pool = mp.Pool(self.core)

        param_dict = dict()

        for i in range(self.core):
            param_dict[str(i)] = [ai[j] for j in range(len(ai)) if j % self.core == i]

        p = [pool.apply_async(self.get_cos_cor,args = (a,)) for a in list(param_dict.values())]

        final_res = [sub_p.get() for sub_p in p]
        return final_res
    
if __name__ == "__main__":
    gwe = get_weighted_edge(6)
    
    data = gwe.get(gwe.final_al_ix)
    pickle.dump(data,open("al_ix.pkl","wb"))