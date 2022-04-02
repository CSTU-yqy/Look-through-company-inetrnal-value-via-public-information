from telnetlib import GA
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

class GAN():

    def __init__(self,thread_num = 4,core = 10,start_year = "1990",end_year = "2021"):
        self.author = "Qianyu Yang"

        self.thread_num = thread_num

        self.core = core

        self.info = pd.read_csv("./data/factor/code_used_name.csv")

        self.got_pool= [i[:-4] for i in os.listdir("./data/key_data")]

        self.target_pool = list(set(self.info.code.tolist()) - set(self.got_pool))

        print(len(self.target_pool))


        self.path = "./chromedriver"

        self.start_year = start_year

        self.end_year = end_year

        self.options = webdriver.ChromeOptions()

        # 谷歌文档提到需要加上这个属性来规避bug
        self.options.add_argument('--disable-gpu') 
        # 不加载图片, 提升速度
        self.options.add_argument('blink-settings=imagesEnabled=false') 
        # 浏览器不提供可视化页面. linux下如果系统不支持可视化不加这条会启动失败
        self.options.add_argument('--headless')


        

    def clean_str(self,text):
        punctuation_str_chinese = punctuation
        punctuation_str_english = string.punctuation
        for i in punctuation_str_english:
            text = text.replace(i, '')
        for i in punctuation_str_chinese:
            text = text.replace(i, '')
        text = re.sub('[a-zA-Z]','',text)
        
        return text

    def get(self,stock_list):
        start_url = "https://kns.cnki.net/KNS8/AdvSearch?dbcode=CCND"
        driver = webdriver.Chrome(executable_path = self.path,options = self.options)
        driver.get(start_url)
        #num_flag = True


        #点击专业检索按钮
        jiansuo_xpath = "//ul[@class='search-classify-menu']/li[@name='majorSearch']"
        driver.find_element_by_xpath(jiansuo_xpath).click()
        time.sleep(1)



        
        for stock_code in stock_list:
            result = pd.DataFrame(columns = ["title","from","date"])
            title = []
            newspaper = []
            date = []
            for stock_name in eval(self.info[self.info.code == stock_code].used_name.iloc[0]):


                #输入检索命令
                
                time.sleep(1)
                search_command = "FT={sn} AND YE>={start_year} AND YE<={end_year}".format(sn = stock_name,start_year = self.start_year,end_year = self.end_year)
                WebDriverWait(driver,3).until(EC.presence_of_element_located((By.XPATH,"/html/body/div[2]/div/div[2]/div/div[1]/div[1]/div[2]/textarea"))).clear()
                driver.find_element_by_xpath("/html/body/div[2]/div/div[2]/div/div[1]/div[1]/div[2]/textarea").send_keys(search_command)
                time.sleep(5)
                

                #点击enter进行搜索
                ActionChains(driver).send_keys(Keys.ENTER).perform()
                time.sleep(1)
                #if num_flag:
                try:
                    WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,"//i[@class='icon icon-sort']"))).click()
                    #driver.find_element_by_xpath("//i[@class='icon icon-sort']").click()
                    sear = driver.find_element_by_xpath("//li[@data-val='50']/a")
                    driver.execute_script("arguments[0].click();",sear)
                    num_flag = False
                except:
                    
                    continue




                #flag标识有没有到达最后一页
                flag = True
                
                page = 0

                while flag:
                    page += 1
                    if page % 3 == 0:
                        time.sleep(10)
                    else:
                        time.sleep(2)
                    try:
                        html_source = driver.page_source
                        label = etree.HTML(html_source)

                        #time.sleep(3)
                        
                        WebDriverWait(driver,5).until(EC.presence_of_all_elements_located((By.XPATH,"//td[@class='date']")))
                        
                        content_text_title = label.xpath("//a[@class='fz14']")
                        content_text_from = label.xpath("//td[@class='source']/a/text()")
                        content_text_time = label.xpath("//td[@class='date']/text()")
                        title_list = [i.xpath("string(.)") for i in content_text_title]
                        
                        title += title_list
                        newspaper += content_text_from
                        date += content_text_time

                        WebDriverWait(driver,3).until(EC.presence_of_element_located((By.XPATH,"//a[@id='PageNext']")))

                        search = driver.find_element_by_xpath("//a[@id='PageNext']")
                        driver.execute_script("arguments[0].click();",search)
                    except selenium.common.exceptions.TimeoutException:
                        flag = False

            result["title"] = title
            result.date = date
            result["from"] = newspaper
            result = result.drop_duplicates()
            result["date"] = result["date"].apply(lambda x:self.clean_str(x))
            result.to_csv("./data/key_data/%s.csv"%stock_code)
        return


    def mp_get(self):
        pool = mp.Pool(self.core)

        param_dict = dict()
        
        for i in range(self.core):
            param_dict[str(i)] = [self.target_pool[j] for j in range(len(self.target_pool)) if j % self.core == i]

        p = [pool.apply_async(self.get,args = (a,)) for a in list(param_dict.values())]

        final_res = [sub_p.get() for sub_p in p]
    
        return 



# we  finally got 4108 company at last
if __name__ == "__main__":

    while len(os.listdir("./data/key_data")) < 4678:
        try:
            GAN().mp_get()
        except:
            time.sleep(60)
            continue 