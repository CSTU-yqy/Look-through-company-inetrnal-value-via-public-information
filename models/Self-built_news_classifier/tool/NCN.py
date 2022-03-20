#!/Users/qianyuyang/opt/anaconda3/bin/python
#@author = Qainyu Yang
#version0.1 --- 20211116

import pandas as pd
import numpy as np
import time
import os
import sys
import jieba
import multiprocessing as mp
import threading
from toolBox import DATA


class NCN():

    def __init__(self):

        self.data = DATA()

        jieba.load_userdict("/Users/qianyuyang/Desktop/paper/data/important/cover_name.txt")

        



    #cut_word是指切分文本的方式，这个版本里面支持三个版本
    #jieba全搜索模式/结合已有的股票曾用名字典和停用词字典的jieba精确搜索模式/分字分字搜索
    def cut_word(self,method,text):

        word_list = list()

        if method == "precise":
            word_list = jieba.cut(text,cut_all=False)

            

        elif method == "global":
            word_list = jieba.cut(text,cut_all=True)

        elif method == "single":
            word_list = list(text)
        
        return [i for i in word_list if i not in self.data.stop_word_list and i not in self.data.cover_name_list]


