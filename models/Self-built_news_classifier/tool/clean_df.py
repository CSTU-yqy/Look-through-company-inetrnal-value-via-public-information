import pandas as pd
import numpy as np
import os
import pickle
import time
import jieba
from bidict import bidict
import warnings
warnings.filterwarnings("ignore")
import re
import multiprocessing as mp

result4 = pd.read_pickle("result4.pkl")
help_analysis1 = result4.groupby("sentence").used_name.apply(set)
help_analysis1 = help_analysis1.reset_index()
def clean(sentence,used_name):
    global help_analysis1
    if len(help_analysis1.loc[help_analysis1["sentence"] == sentence,"used_name"].iloc[0]) > 1:
        
        # This case description has other distractors
        gr = help_analysis1.loc[help_analysis1["sentence"] == sentence,"used_name"].iloc[0] - set(used_name)
        new_sentence = []
        for s in sentence.split("，"):
            if used_name in s:
                
                #If the short_name of this sentence is there, remove the names of other distractors
                for _gr in gr:
                    s = s.replace(_gr,"")
                    
            else:
                
                for _gr in gr:
                    if _gr in s:
                        s = ""
                        break
            new_sentence.append(s)
        return ",".join(new_sentence)
    else:
        
        return sentence

    def clean_df(df):
        df["sentence"] = df.apply(lambda x:clean(x["sentence"],x["used_name"]),axis = 1)
        return df