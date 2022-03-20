# coding: UTF-8
import time

from sympy import frac
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
import pandas as pd
import copy
from random import choice
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from random import choice
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument("--batch_size",type = int,required = True,help = "set the batch size of mini batch")
parser.add_argument("--pad_size",type = int,required = True,help = "set the pad size of each text")
args = parser.parse_args()


if __name__ == '__main__':

    
    """
    数据加工
    """
    """
    验证集和测试集要同分布
    """
    sample_data = pd.read_pickle("F:\paper\code\getCnki\handle_data\single_file\\new_predict\judge good bad news bert\sample_data.pkl")
    sample_data.label = sample_data.label.map({-1:0,1:1})
    _train,_test = train_test_split(sample_data,train_size = 0.6)
    _test,_dev = train_test_split(_test,train_size = 0.5)
    _test.to_pickle("predict/test.pkl")
    _train_good = _train[~_train.code.isin(_train[_train.label == 0].code.tolist())]
    _train_bad = _train[_train.code.isin(_train[_train.label == 0].code.tolist())]
    print(len(_train))
    for i in range(len(_train_bad)):
        
        for j in range(10):
            dat = _train_bad.iloc[i]
            dat["model_sentence"] = dat["model_sentence"] + "，" + choice(_train_good.sample(1)["model_sentence"].iloc[0].split("，"))
            _train = _train.append(dat)
    print(len(_train))
    _train = _train.sample(frac = 1)


    _dev_good = _dev[~_dev.code.isin(_train[_train.label == 0].code.unique().tolist())]
    _dev_bad = _dev[_dev.code.isin(_train[_train.label == 0].code.unique().tolist())]
    #print(len(_train))
    for i in range(len(_dev_bad)):
        
        for j in range(10):
            dat = _dev_bad.iloc[i]
            dat["model_sentence"] = dat["model_sentence"] + "，" + choice(_dev_good.sample(1)["model_sentence"].iloc[0].split("，"))
            _dev = _dev.append(dat)
    #print(len(_train))
    _dev = _dev.sample(frac = 1)

    _test_good = _test[~_test.code.isin(_train[_train.label == 0].code.unique().tolist())]
    _test_bad = _test[_test.code.isin(_train[_train.label == 0].code.unique().tolist())]
    #print(len(_train))
    for i in range(len(_test_bad)):
        
        for j in range(10):
            dat = _test_bad.iloc[i]
            dat["model_sentence"] = dat["model_sentence"] + "，" + choice(_test_good.sample(1)["model_sentence"].iloc[0].split("，"))
            _test = _test.append(dat)
    #print(len(_train))
    _test = _test.sample(frac = 1)
    
    for dd in ["_train","_test","_dev"]:
        with open("THUCNews\data\%s.txt"%dd[1:],"w",encoding = "utf-8",errors = "ignore") as f:
            ddd = eval(dd)
            for i in range(len(ddd)):
                sentence = ddd.model_sentence.iloc[i]
                label = ddd.label.iloc[i]
                f.write("%s\t%s\n"%(sentence,label))
            f.close()

    """
    数据加工结束
    """
    print("finish")




    dataset = 'THUCNews'  # 数据集

    model_name = args.model  # bert
    batch_size  = args.batch_size
    pad_size = args.pad_size
    x = import_module('models.' + model_name)
    config = x.Config(dataset,batch_size,pad_size)
    # config.batch_size = batch_size
    # config.pad_size = pad_size
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
