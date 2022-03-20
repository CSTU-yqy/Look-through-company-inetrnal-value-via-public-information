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

class GNU():
    def __init__(self):
        self.author = "Qainyu Yang"
        self.start_url = "https://navi.cnki.net/knavi/newspapers/index"
        self.path = "/Users/qianyuyang/Desktop/paper/code/chromedriver"
        self.driver = webdriver.Chrome(executable_path = self.path)
        self.url_dict = dict()
        self.xpath_hash = {
            "zyj":"/html/body/div[2]/div/div[2]/div[1]/div[1]/ul/li[1]/span/a",
            "dfj":"/html/body/div[2]/div/div[2]/div[1]/div[1]/ul/li[2]/span/a",
            "lb":"/html/body/div[2]/div/div[2]/div[2]/div/div[1]/ul/li[2]/a",
            "news_button":"//span[@class='tab_1']/h2/a",
            "next_page":"/html/body/div[2]/div/div[2]/div[2]/div/div[3]/a[contains(@class,'next')]"
        }
        self.driver.get(self.start_url)

    def transform_url(self,newspaper_name):
        return "https://navi.cnki.net/knavi/newspapers/%s/detail?uniplatform=NZKPT"%newspaper_name


    @wraps
    def get_url(self):

        for input_xpath in [self.xpath_hash["zyj"],self.xpath_hash["dfj"]]:
            WebDriverWait(self.driver,20).until(EC.presence_of_all_elements_located((By.XPATH,input_xpath)))[0].click()
            time.sleep(5)
            WebDriverWait(self.driver,20).until(EC.presence_of_all_elements_located((By.XPATH,self.xpath_hash["lb"])))[0].click()
            time.sleep(5)
            
            WebDriverWait(self.driver,20).until(EC.presence_of_all_elements_located((By.XPATH,self.xpath_hash["news_button"])))
            html_source = self.driver.page_source
            label = etree.HTML(html_source)
            content_url = label.xpath(self.xpath_hash["news_button"] + "/@href")
            content_text = label.xpath(self.xpath_hash["news_button"] + "/text()")
            for i in range(len(content_text)):
                self.url_dict[content_text[i]] = self.transform_url(content_url[i][-4:])

            try:
                while True:
                    WebDriverWait(self.driver,20).until(EC.presence_of_element_located((By.XPATH,self.xpath_hash["next_page"]))).click()
                    time.sleep(5)
                    WebDriverWait(self.driver,20).until(EC.presence_of_all_elements_located((By.XPATH,self.xpath_hash["news_button"])))
                    html_source = self.driver.page_source
                    label = etree.HTML(html_source)
                    content_url = label.xpath(self.xpath_hash["news_button"] + "/@href")
                    content_text = label.xpath(self.xpath_hash["news_button"] + "/text()")
                    for i in range(len(content_text)):
                        self.url_dict[content_text[i]] = self.transform_url(content_url[i][-4:])

            except:

                with open('url_dict.json','w') as file:
                    json.dump(self.url_dict,file)
                    file.close()
                continue
        return self.url_dict

        
class GNT():

    def __init__(self,thread_num = 4,core = 6,stock_list = pd.read_pickle("stock_name.pkl").tolist()):
        self.author = "Qianyu Yang"

        self.thread_num = thread_num

        self.core = core

        with open("url_dict_sj.json","r") as f:
            self.info = json.load(f)
            f.close()


        self.stock_list = stock_list

        self.news_stat = dict()

        self.result = pd.DataFrame(
            index = self.info.keys(),
            columns = self.stock_list.keys()
            )
        
        self.path = "/Users/qianyuyang/Desktop/paper/code/chromedriver"

        self.options = webdriver.ChromeOptions()

        self.options.add_argument('--disable-gpu') 
 
        self.options.add_argument('blink-settings=imagesEnabled=false') 
        
        self.options.add_argument('--headless') 

        self.driver = webdriver.Chrome(executable_path = self.path,chrome_options = self.options)

        self.xpath_hash = {
            "news_num":"//p[@class='hostUnit']/label[contains(text(),'文献篇数')]/../span/text()",
            "gxk":"/html/body/div[2]/div[2]/div[3]/div/ul[2]/div[1]/select",
            "qw":"/html/body/div[2]/div[2]/div[3]/div/ul[2]/div[1]/select/option[4]",
            "search_block":"//input[@id='J_searchTxt']",
            "search_button":"//a[@class='btn-search']",
            "news":"//td[@class='name']/a[@target='_blank']",
            "target_news_num":"//span[@id='partiallistcount']/text()",
            "target_news":"//td[@class='name']/a",
            "next_page":"//a[@class='next']",
            "cannot_find_signal":'//div[@style="font-size:20px; text-align:center;line-height: 100px;min-height: 100px;"]'
        }

    def clean_str(text):
        punctuation_str_chinese = punctuation
        punctuation_str_english = string.punctuation
        for i in punctuation_str_english:
            text = text.replace(i, '')
        for i in punctuation_str_chinese:
            text = text.replace(i, '')
        text = re.sub('[a-zA-Z]','',text)
        
        return text


    def get(self,key_list,stock_dict):
    
        for key in key_list:
            url = self.info[key]


            driver = webdriver.Chrome(executable_path = self.path,chrome_options = self.options)
            driver.get(url)
            WebDriverWait(driver,20).until(EC.presence_of_element_located((By.XPATH,self.xpath_hash["gxk"]))).click()
            
            html_source = driver.page_source
            label = etree.HTML(html_source)
            content_text = label.xpath(self.xpath_hash["news_num"])
            self.news_stat[key] = content_text[0]

            with open('news_stat.json','w') as file:
                json.dump(self.news_stat,file)
                file.close()
            time.sleep(1)
            WebDriverWait(driver,20).until(EC.presence_of_element_located((By.XPATH,self.xpath_hash["qw"]))).click()


            for stock_name in stock_dict.index:
                res = 0
                stock_list = stock_dict.loc[stock_name]
                for stock in stock_list:


                    WebDriverWait(driver,20).until(EC.presence_of_element_located((By.XPATH,self.xpath_hash["search_block"]))).clear()


                    WebDriverWait(driver,20).until(EC.presence_of_element_located((By.XPATH,self.xpath_hash["search_block"]))).send_keys(stock)

                    t1 = time.time()


                    WebDriverWait(driver,20).until(EC.presence_of_element_located((By.XPATH,self.xpath_hash["search_button"])))

                    searcher = driver.find_element_by_xpath(self.xpath_hash["search_button"])

                    t2 = time.time()
                    time.sleep(max(1 - t2 + t1,0))
                    driver.execute_script("arguments[0].click();", searcher)

                    try:
                        time.sleep(1)
                        driver.find_element_by_xpath(self.xpath_hash["cannot_find_signal"])
                        content_text = 0

                    except:
                        try:
                        
                            WebDriverWait(driver,5).until(EC.presence_of_element_located((By.XPATH,self.xpath_hash["target_news"])))
                            
                            
                            html_source = driver.page_source
                            label = etree.HTML(html_source)
                            content_text = label.xpath(self.xpath_hash["target_news_num"])[0]
                        except:
                            content_text = 0
                    
                    res += int(content_text)



                self.result.loc[key,stock_name] = res
            driver.close()

            self.result.to_csv("newspaper_comapny_stat.csv")

    def thread_get(self,key_list):

        threads = []
        param_dict = dict()
        for i in range(self.thread_num):
            param_dict[str(i)] = self.stock_list.iloc[[j for j in range(len(self.stock_list)) if j % self.thread_num == i]]

        for sub_stock_list in param_dict.values():
            thread = Thread(target = self.get,args = (key_list,sub_stock_list))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()



    def get_the_world(self):

        pool = mp.Pool(self.core)

        param_dict = dict()
        
        for i in range(self.core):
            param_dict[str(i)] = [list(self.info.keys())[j] for j in range(len(self.info.keys())) if j % self.core == i]

        p = [pool.apply_async(self.thread_get,args = (a,)) for a in list(param_dict.values())]

        for sub_p in p:
            sub_p.get()
     
        return self.result


    def mp_get(self,stock_list):
        pool = mp.Pool(self.core)

        param_dict = dict()
        
        for i in range(self.core):
            param_dict[str(i)] = [list(self.info.keys())[j] for j in range(len(self.info.keys())) if j % self.core == i]

        p = [pool.apply_async(self.get,args = (a,stock_list,)) for a in list(param_dict.values())]

        
        final_res = [sub_p.get() for sub_p in p]
       

        
        return self.result




class GHTN():
    
    def __init__(self,thread_num = 4,core = 11,wait = 7200):
        self.author = "Qianyu Yang"

        self.thread_num = thread_num

        self.core = core

        self.data = pd.read_pickle("C:\\Users\\用户1\\Desktop\\pp\\code\\news_hash.pkl")
        
        self.target_index_list = list(set(self.data.index) - set([int(i[:-4]) for i in os.listdir("C:\\Users\\用户1\\Desktop\\pp\\cnki_text") if "_" not in i]))

        self.path = "C:\\Users\\用户1\\Desktop\\pp\\code\\chromedriver.exe"

        self.options = webdriver.ChromeOptions()

        # the google offical document said we need add the following code to avoid bug
        self.options.add_argument('--disable-gpu') 
        
        # Do not load pictures, improve speed
        self.options.add_argument('blink-settings=imagesEnabled=false') 
        
        # The browser does not provide visualization pages. If the system does not support visualization under linux, it will fail to start.
        #self.options.add_argument('--headless')

        self.start_time = time.time()

        self.wait = wait

    def clean_str(self,text):
        punctuation_str_chinese = punctuation
        punctuation_str_english = string.punctuation
        for i in punctuation_str_english:
            text = text.replace(i, '')
        for i in punctuation_str_chinese:
            text = text.replace(i, '')
        text = re.sub('[a-zA-Z]','',text)
        
        return text

    def get(self,sub_index_list):
        start_url = "https://kns.cnki.net/KNS8/AdvSearch?dbcode=CCND"
        driver = webdriver.Chrome(executable_path = self.path,options = self.options)
        driver.get(start_url)
        
        #click the search button
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
            
            #input our search command
            time.sleep(1)
            search_command = "TI={ti} AND LY={nwp}".format(ti = name,nwp = newspaper)
            try:
                WebDriverWait(driver,10,1).until(EC.presence_of_element_located((By.XPATH,"/html/body/div[2]/div/div[2]/div/div[1]/div[1]/div[2]/textarea"))).clear()
                driver.find_element_by_xpath("/html/body/div[2]/div/div[2]/div/div[1]/div[1]/div[2]/textarea").send_keys(search_command)
            except:
                continue
            result = pd.DataFrame(columns = ["title","from","date"])

            #click enter
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

                    with open("C:\\Users\\用户1\\Desktop\\pp\\cnki_text\\{n}.txt".format(n = i),"w") as f:
                        f.write(text)
                        f.close()

                    url = driver.current_url

                    with open("C:\\Users\\用户1\\Desktop\\pp\\cnki_url\\{n}.txt".format(n = i),"w") as f:
                        f.write(url)
                        f.close()
                
            except:
                pass
            driver.close()
            driver.switch_to.window(grand_window)

            time.sleep(5)


        
        return


    def mp_get(self,device_num_max,device_num_min):
        """
        _description_
            
        Because we need to handle large amount of data, 
        we generally deploy the program in multiple servers in parallel, 
        each server is responsible for the data of one of the index ranges, 
        'device_num_max' & 'device_num_min' helps us to control the tasks between different servers do not overlap
        """
        pool = mp.Pool(self.core)

        param_dict = dict()
        
        for i in range(self.core):
            param_dict[str(i)] = [self.target_index_list[j] for j in range(len(self.target_index_list)) if j % self.core == i and self.target_index_list[j] >= device_num_max and self.target_index_list[j] <= device_num_min]

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
    
            



    


        
        





