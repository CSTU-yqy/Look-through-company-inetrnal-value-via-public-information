import selenium
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import os
import pandas as pd
import numpy as np
import time
import datetime
import sys
#need attention
import multiprocessing.dummy as mp
from lxml import etree
import requests
import logging
import pickle
import json
import string
from zhon.hanzi import punctuation
from threading import Thread
from functools import wraps
import re
import warnings
warnings.filterwarnings("ignore")
class GHTN():

    def __init__(self,thread_num = 4,core = 12,wait = 7200,up_limit = 200000,low_limit = 0):
        self.author = "Qianyu Yang"

        self.thread_num = thread_num

        self.core = core

        self.data = pd.read_pickle("./data/news_hash.pkl")

        
        self.target_index_list = list(set(self.data.index) - set([int(i[:-4]) for i in os.listdir("./data/cnki_text") if "_" not in i]))

        self.up_limit = up_limit

        self.low_limit = low_limit

        self.path = "./chromedriver.exe"



        self.options = webdriver.ChromeOptions()

        # 谷歌文档提到需要加上这个属性来规避bug
        self.options.add_argument('--disable-gpu') 
        # 不加载图片, 提升速度
        self.options.add_argument('blink-settings=imagesEnabled=false') 
        # 浏览器不提供可视化页面. linux下如果系统不支持可视化不加这条会启动失败
        self.options.add_argument('--headless')

        self.start_time = time.time()

        self.wait = wait

        self.name_list = self.data["title"].tolist()

        self.newspaper_list = self.data["from"].tolist()


    @staticmethod
    def clean_key_data():
        file_path = "./data/key_data"
        data_pool = []
        for file in os.listdir(file_path):
            data = pd.read_csv(os.path.join(file_path,file))
            data_pool.append(data)
        data = pd.concat(data_pool)
        data = data.drop("Unnamed: 0",axis = 1).drop_duplicates().dropna()
        data = data.reset_index(drop = True)
        data.to_pickle("./data/news_hash.pkl")
        return data



    def clean_str(self,text):
        punctuation_str_chinese = punctuation
        punctuation_str_english = string.punctuation
        for i in punctuation_str_english:
            text = text.replace(i, '')
        for i in punctuation_str_chinese:
            text = text.replace(i, '')
        text = re.sub('[a-zA-Z]','',text)
        
        return text
#####
    def get(self,sub_index_list):
        start_url = "https://kns.cnki.net/KNS8/AdvSearch?dbcode=CCND"
        driver = webdriver.Chrome(executable_path = self.path,options = self.options)
        driver.get(start_url)
        #num_flag = True


        #点击专业检索按钮
        jiansuo_xpath = "//ul[@class='search-classify-menu']/li[@name='majorSearch']"
        driver.find_element_by_xpath(jiansuo_xpath).click()
        time.sleep(1)
        grand_window = driver.window_handles[0]
        
        flag = 0


        for i in sub_index_list:

            current_time= time.time()
            if current_time - self.start_time >= self.wait:
                driver.quit()
                break
            
            flag += 1
            if flag % 100 == 0:
                time.sleep(60)
            name = self.data.title.loc[i]
            newspaper = self.data["from"].iloc[i]
            # name = self.name_list[i]
            # newspaper = self.newspaper_list[i]
            #输入检索命令
            time.sleep(1)
            search_command = "TI={ti} AND LY={nwp}".format(ti = name,nwp = newspaper)
            try:
                WebDriverWait(driver,10,1).until(EC.presence_of_element_located((By.XPATH,"/html/body/div[2]/div/div[2]/div/div[1]/div[1]/div[2]/textarea"))).clear()
                driver.find_element_by_xpath("/html/body/div[2]/div/div[2]/div/div[1]/div[1]/div[2]/textarea").send_keys(search_command)
            except:
                continue
            result = pd.DataFrame(columns = ["title","from","date"])

            #点击enter进行搜索
            ActionChains(driver).send_keys(Keys.ENTER).perform()
            time.sleep(1)
            try:
                WebDriverWait(driver,3,1).until(EC.presence_of_element_located((By.XPATH,"//a[@onclick='return disableButton2(this);']"))).click()
                
            except:
                pass

            
            try:
                WebDriverWait(driver,10,2).until(EC.presence_of_element_located((By.XPATH,"//a[@class='icon-html']"))).click()
                time.sleep(1)
            except:
                continue
            
            try:
                #####
                driver.switch_to.window(driver.window_handles[-1])
                WebDriverWait(driver,3,1).until(EC.presence_of_element_located((By.XPATH,"//a[@onclick='return disableButton2(this);']"))).click()
                time.sleep(1)
            except:
                driver.switch_to.window(driver.window_handles[-1])
                time.sleep(1)
            
            
            try:
                WebDriverWait(driver,5).until(EC.presence_of_all_elements_located((By.XPATH,"//div[@class='p1']/p")))
                html_source = driver.page_source
                label = etree.HTML(html_source)
                context = label.xpath("//div[@class='p1']/p/text()")
                if len(context) != 0:
                    text = ("\r").join(context)

                    with open("./data/cnki_text/{n}.txt".format(n = i),"w") as f:
                        f.write(text)
                        f.close()

                    url = driver.current_url

                    with open("./data/cnki_url/{n}.txt".format(n = i),"w") as f:
                        f.write(url)
                        f.close()
                
            except:
                pass
            driver.close()
            driver.switch_to.window(grand_window)

            time.sleep(5)


        
        return


    def mp_get(self):
        pool = mp.Pool(self.core)

        param_dict = dict()
        
        for i in range(self.core):
            param_dict[str(i)] = [self.target_index_list[j] for j in range(len(self.target_index_list)) if j % self.core == i and self.target_index_list[j] >= self.low_limit and self.target_index_list[j] <= self.up_limit]

        p = [pool.apply_async(self.get,args = (a,)) for a in list(param_dict.values())]

        final_res = [sub_p.get() for sub_p in p]
    
        return


if __name__ == "__main__":
    while True:

        try:
            GHTN(wait = 7200).mp_get()
            time.sleep(600)
        except:
            pass
    