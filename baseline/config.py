import torch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')



class Config(object):
    def __init__(self):
        self.bert_cache_path = '/home/tzh/model/bert-base-chinese'
        # self.bert_cache_path = 'hfl/chinese-roberta-wwm-ext'
        # self.train_dataset_path = "/home/tzh/BRC/data/CHEF新/CHEF_train_简.json"
        # self.test_dataset_path = "/home/tzh/BRC/data/CHEF新/CHEF_test_简.json"
        self.train_dataset_path = "/home/tzh/BRC/data/CHEF/train_简_召回.json"
        self.dev_dataset_path = "/home/tzh/BRC/data/CHEF/dev_简_召回.json"
        self.test_dataset_path = "/home/tzh/BRC/data/CHEF/test_简_召回.json"
        # self.dev_dataset_path = 
        # self.train_dataset_path = "Data/train.json"
        # self.test_dataset_path = "Data/test.json"
        
        # hyper parameter
        self.num_classes = 3
        self.epochs = 30
        self.batch_size = 1
        self.lr = 1e-5
        self.tuning_bert_rate = 1e-5
        self.gradient_accumulation_steps = 2
        self.dp = 0.1
        self.warmup_proportion = 0.1

        # gnn
        self.feat_dim = 768
        self.gnn_dims = '192'
        self.att_heads = '4'
