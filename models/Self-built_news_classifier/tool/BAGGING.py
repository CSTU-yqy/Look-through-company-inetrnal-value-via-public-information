import pandas as pd
from sklearn.model_selection import KFold

class BAGGING():
    def __init__(self,sample_size = 5,org_address = "F:/paper/data/important/origin_sample_data.pkl",bad_news_address = []):
        self._author = "Qianyu Yang"
        self._sample_size = sample_size
        #self._sample = sample
        self._org = pd.read_pickle(org_address)
        self._bad_news_data = pd.concat([pd.read_excel(i).rename(columns = {"Unnamed: 0":"id"}) if "id" not in pd.read_excel(i).columns else pd.read_excel(i) for i in bad_news_address]) if bad_news_address else pd.DataFrame()
        self._all_sample_data = pd.concat([self._org,self._bad_news_data]).drop("Unnamed: 0",axis = 1) if "Unnamed: 0" in self._bad_news_data.columns else pd.concat([self._org,self._bad_news_data]).drop_duplicates("sentence")
        self._bad_sample_data = self._all_sample_data[self._all_sample_data.label == -1]
        self._good_sample_data = self._all_sample_data[self._all_sample_data.label == 1]
    @property
    def data(self):
        data = {
            "train_data":[],
            "test_data":[]
        }
        for i in range(self._sample_size):
            bad_train_data = self._bad_sample_data.sample(len(self._bad_sample_data),replace = True)
            good_train_data = self._good_sample_data.sample(len(self._good_sample_data),replace = True)
            train_data = pd.concat([bad_train_data,good_train_data])
            train_data = train_data.drop_duplicates()
            test_data = self._all_sample_data[self._all_sample_data.id.isin(train_data.id) == False]
            test_data = test_data.drop_duplicates()
            data["train_data"].append(train_data)
            data["test_data"].append(test_data)
        return data

class kfold():
    def __init__(self,data,n_splits):
        self.author = "Qianyu Yang"
        self.data = data
        self.n_splits = n_splits
        self.train_set = []
        self.test_set = []
    
    @property
    def train_data(self):
        kf = KFold(n_splits = self.n_splits,shuffle = True)
        for train_index,test_index in kf.split(self.data):
            self.train_set.append(self.data.iloc[train_index])
            self.test_set.append(self.data.iloc[test_index])
        return self.train_set
    
    # @train_data.setter
    # def train_data(self):
    #     kf = KFold(n_splits = self.n_splits)
    #     for train_index,test_index in kf.split(self.data):
    #         self.train_set.append(self.data.iloc[train_index])
    #         self.test_set.append(self.data.iloc[test_index])
        
    @property
    def test_data(self):
        return self.test_set

