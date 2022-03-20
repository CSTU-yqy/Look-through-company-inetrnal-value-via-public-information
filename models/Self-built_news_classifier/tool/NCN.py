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

        



    #cut_word refers to the way to cut text, this version supports three versions
    #jieba full search mode / jieba exact search mode combined with existing stock used name dictionary and stop word dictionary / word-by-word search
    def cut_word(self,method,text):

        word_list = list()

        if method == "precise":
            word_list = jieba.cut(text,cut_all=False)

            

        elif method == "global":
            word_list = jieba.cut(text,cut_all=True)

        elif method == "single":
            word_list = list(text)
        
        return [i for i in word_list if i not in self.data.stop_word_list and i not in self.data.cover_name_list]


