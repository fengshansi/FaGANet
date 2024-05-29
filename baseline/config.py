import torch
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

TORCH_SEED = 129


class Config(object):
    def __init__(self):
        self.bert_cache_path = 'bert-base-chinese'
        # self.bert_cache_path = 'hfl/chinese-roberta-wwm-ext'
        self.train_dataset_path = "/data/lwy/Projects/CHEF/CHEF7.7/data/CHEF_train.json"
        self.test_dataset_path = "/data/lwy/Projects/CHEF/CHEF7.7/data/CHEF_test.json"
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
