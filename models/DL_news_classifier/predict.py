import torch
from importlib import import_module
import time
from train_eval import predict
from utils import build_predict_dataset,build_iterator
import argparse
import pickle
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: bert, ERNIE, bert_RCNN.....')
parser.add_argument('--data_path', type=str, required=True, help='where the test data store')
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--pad_size', type=int, required=True)
args = parser.parse_args()
model_name = args.model  # bert
batch_size = args.batch_size
pad_size = args.pad_size
data_path = args.data_path
if __name__ == "__main__":
    model_store = "F:\paper\code\getCnki\handle_data\single_file\clean_data_2\\bert\classify company news bert\THUCNews\saved_dict"
    x = import_module('models.' + model_name)
    config = x.Config('THUCNews')
    test_data_iter = build_iterator(build_predict_dataset(config,data_path),config)
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(model_store + "\%s_%s_%s.ckpt"%(model_name,batch_size,pad_size)))
    file = open("predict\%s_%s_%s.pkl"%(model_name,batch_size,pad_size),"wb")
    pickle.dump(predict(config,model,test_data_iter),file)
    file.close()