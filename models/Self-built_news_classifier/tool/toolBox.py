import pandas as pd
import numpy as np
import sys
sys.path.append("F:\paper\code\getCnki\handle_data")
import copy




class DATA():
    def __init__(self):
        self.stock_name_info = pd.read_pickle("F:/paper/data/important/all_stock_name_info.pkl")

        self.news_hash = pd.read_pickle("F:/paper/data/important/news_hash.pkl")

        with open("F:/paper/data/important/cover_name.txt","r",encoding = "gbk",errors = "ignore") as f1:
            self.cover_name_list = f1.read().split("\n")

        with open("F:/paper/data/important/stopwords.txt","r",encoding = "gbk",errors = "ignore") as f2:
            self.stop_word_list = f2.read().split("\n")
    """
    这是一个专门的清晰抽样数据的工具，例如清洗org,test_data

    """
from tool.get_different_mindf import GetData
class TOOl():
    def __init__(self):
        self.autor = "Qainyu Yang"

    
    def clean(self,org,min_df):
        gd = GetData(min_df = min_df,cut_method="global",all_data_address = "F:\paper\code\getCnki\handle_data\single_file\clean_data_2\data\\all_data_cut_global.pkl")



        all_data = gd.data
    
        all_data = all_data.drop_duplicates(["sentence"])
        all_data["id"] = all_data.index
        ad = copy.deepcopy(all_data)
        org = org.drop(['text'],axis = 1) if "text" in org.columns else org
        
        if "id" not in org.columns:
            org = org.rename(columns = {"Unnamed: 0":"id"})
        org = org.drop("id",axis = 1)
        #test_data = test_data.drop("id",axis = 1)
        org = pd.merge(org,ad[["sentence","text","id"]],on = "sentence",how = "left")
        return org
