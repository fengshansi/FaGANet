import os
import csv
from config import *
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertForSequenceClassification,
)
from model_baseline import Network
import datetime
import numpy as np
import pandas as pd
import pickle

from torcheval.metrics import functional as FUNC
import argparse
parser = argparse.ArgumentParser()
# nargs="+" 支持多个参数 且解析参数的返回值默认为list
parser.add_argument("--TORCH_SEED", type=str, default=0)
args = parser.parse_args()

TORCH_SEED = args.TORCH_SEED

CACHE_DIR = "/home/tzh/FaGANet/from_pretrained"


# dataset
class myDataset(Dataset):
    def __init__(self, tokenizer, path, configs):
        self.tokenizer = tokenizer
        self.data_path = path
        self.num_classes = configs.num_classes

        self.claim = []
        self.claimId = []
        self.context = []
        self.word_count = []
        self.doc_len = []
        self.clause_len = []
        self.label = []
        self.discourse = []
        self.discourse_mask = []
        self.segment_mask = []
        self.query_len = []
        self.discourse_adj = []
        self.ans_mask = []

        df = pd.read_json(self.data_path)

        for i in range(len(df)):
            claim = df["claim"][i]
            claimId = int(df["claimId"][i])
            # 不使用召回分数 只使用召回句子
            raw_context = [tmp[0] for tmp in df["dense_retrieval"][i]]
            label = int(df["label"][i])

            # Preprocess
            # Set length threshold 少于  80
            if len(claim) > 80:
                claim = claim[:80]
            for j in range(len(raw_context)):
                if len(raw_context[j]) > 80:
                    raw_context[j] = raw_context[j][:80]

            context = ""
            word_count = 0  # 五条的总字数
            clause_len = []  # 五条的每一句的长度
            doc_len = len(raw_context)
            # 设置了小于5的情况
            while doc_len < 5:
                raw_context.append("")
                doc_len += 1
            for j in range(doc_len):
                context += raw_context[j]
                clause_len.append(len(raw_context[j]))
                word_count += len(raw_context[j])

            discourse = "[CLS]" + claim + "[SEP]" + context + "[SEP]"
            discourse = torch.Tensor(
                tokenizer(discourse, padding="max_length", max_length=512)["input_ids"]
            ).to(torch.int32)
            discourse_mask = torch.Tensor(
                [1] * (len(claim) + word_count + 3)
                + [0] * (512 - (len(claim) + word_count + 3))
            ).to(torch.int32)
            segment_mask = torch.Tensor(
                [0] * (len(claim) + 2) + [1] * (512 - (len(claim) + 2))
            ).to(torch.int32)

            query_len = len(claim) + 2
            discourse_adj = torch.ones([doc_len, doc_len])
            label = torch.tensor(label)

            # 看是两个二分类还是三分类
            label = torch.nn.functional.one_hot(label, num_classes=self.num_classes)
            ans_mask = torch.ones(self.num_classes)

            self.claim.append(claim)
            self.claimId.append(claimId)
            self.context.append(context)
            self.word_count.append(word_count)
            self.doc_len.append(doc_len)
            self.clause_len.append(clause_len)
            self.label.append(label)
            self.discourse.append(discourse)
            self.discourse_mask.append(discourse_mask)
            self.segment_mask.append(segment_mask)
            self.query_len.append(query_len)
            self.discourse_adj.append(discourse_adj)
            self.ans_mask.append(ans_mask)

    def __getitem__(self, item):
        return (
            self.claim[item],
            self.claimId[item],
            self.context[item],
            self.word_count[item],
            self.doc_len[item],
            self.clause_len[item],
            self.label[item],
            self.discourse[item],
            self.discourse_mask[item],
            self.segment_mask[item],
            self.query_len[item],
            self.discourse_adj[item],
            self.ans_mask[item],
        )

    def __len__(self):
        return len(self.claim)


# evaluate one batch
def evaluate_one_batch(configs, batch, model, tokenizer):

    batch_size = configs.batch_size

    (
        claim,
        claimId,
        context,
        word_count,
        doc_len,
        clause_len,
        label,
        discourse,
        discourse_mask,
        segment_mask,
        query_len,
        discourse_adj,
        ans_mask,
    ) = batch

    pred = model(
        discourse,
        discourse_mask,
        segment_mask,
        query_len,
        clause_len,
        doc_len,
        discourse_adj,
        label,
    )
    true = label

    pred = torch.argmax(pred).to("cpu")
    true = torch.argmax(true).to("cpu")

    pred, true = int(pred), int(true)
    return pred, true


# evaluate step
def evaluate(configs, test_loader, model, tokenizer):
    NC = configs.num_classes
    model.eval()
    pred_list = []
    true_list = []
    for batch in test_loader:
        pred, true = evaluate_one_batch(configs, batch, model, tokenizer)
        pred_list.append(pred)
        true_list.append(true)
    pred_list = torch.tensor(pred_list)
    true_list = torch.tensor(true_list)

    # Micro和Macro结果
    micro_result = [
        FUNC.multiclass_f1_score(pred_list, true_list, average="micro", num_classes=NC),
        FUNC.multiclass_recall(pred_list, true_list, average="micro", num_classes=NC),
        FUNC.multiclass_precision(
            pred_list, true_list, average="micro", num_classes=NC
        ),
    ]

    macro_result = [
        FUNC.multiclass_f1_score(pred_list, true_list, average="macro", num_classes=NC),
        FUNC.multiclass_recall(pred_list, true_list, average="macro", num_classes=NC),
        FUNC.multiclass_precision(
            pred_list, true_list, average="macro", num_classes=NC
        ),
    ]

    # 计算每个标签的f1分数
    per_class_f1 = FUNC.multiclass_f1_score(
        pred_list, true_list, average=None, num_classes=NC
    ).tolist()

    # 拼接macro结果和每个标签的f1分数
    macro_result = macro_result + per_class_f1

    return micro_result, macro_result


def main(configs, train_loader, dev_loader, test_loader, tokenizer):
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True

    # model
    model = Network(configs).to(DEVICE)
    # optimizer
    params = list(model.named_parameters())
    optimizer_grouped_params = [
        {"params": [p for n, p in params if "_bert" in n], "weight_decay": 0.01},
        {
            "params": [p for n, p in params if "_bert" not in n],
            "lr": configs.lr,
            "weight_decay": 0.01,
        },
    ]
    optimizer = AdamW(params=optimizer_grouped_params, lr=configs.tuning_bert_rate)

    # scheduler
    training_steps = (
        configs.epochs * len(train_loader) // configs.gradient_accumulation_steps
    )
    warmup_steps = int(training_steps * configs.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
    )

    # training
    model.zero_grad()
    early_stop_flag = 0
    max_dev_micro_result = None
    max_dev_macro_result = None

    the_test_micro_result = None
    the_test_macro_result = None

    for epoch in range(1, configs.epochs + 1):
        for train_step, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            batch_size = configs.batch_size
            (
                claim,
                claimId,
                context,
                word_count,
                doc_len,
                clause_len,
                label,
                discourse,
                discourse_mask,
                segment_mask,
                query_len,
                discourse_adj,
                ans_mask,
            ) = batch
            pred = model(
                discourse,
                discourse_mask,
                segment_mask,
                query_len,
                clause_len,
                doc_len,
                discourse_adj,
                label,
            )

            loss = model.loss_pre(pred, label, ans_mask)
            loss.backward()

            if train_step % configs.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if train_step % 1000 == 0:
                print(
                    "epoch: {}, step: {}, loss: {}".format(epoch, train_step, loss),
                    flush=True,
                )

        with torch.no_grad():
            dev_micro_result, dev_macro_result = evaluate(
                configs, dev_loader, model, tokenizer
            )
            test_micro_result, test_macro_result = evaluate(
                configs, test_loader, model, tokenizer
            )

            # 按micro来选 原先这里居然是分着找最大micor和macro的 离谱 两个都不对着同一个模型
            if (
                max_dev_micro_result is None
                or dev_micro_result[0] > max_dev_micro_result[0]
            ):
                early_stop_flag = 1
                max_dev_micro_result = dev_micro_result
                max_dev_macro_result = dev_macro_result
                the_test_micro_result = test_micro_result
                the_test_macro_result = test_macro_result

                state_dict = {
                    "model": model.state_dict(),
                    "result": max_dev_micro_result,
                }
                torch.save(state_dict, "model/model.pth")

            else:
                early_stop_flag += 1

            print(
                f"max_dev_micro_result: {max_dev_micro_result}, max_dev_macro_result: {max_dev_macro_result}\nthe_test_micro_result: {the_test_micro_result},\nthe_test_macro_result: {the_test_macro_result}"
            )

        if early_stop_flag >= 10:
            break

    return max_dev_micro_result, max_dev_macro_result


def my_collate_fn(batch):
    configs = Config()
    batch_size = configs.batch_size

    batch = zip(*batch)
    (
        claim,
        claimId,
        context,
        word_count,
        doc_len,
        clause_len,
        label,
        discourse,
        discourse_mask,
        segment_mask,
        query_len,
        discourse_adj,
        ans_mask,
    ) = batch

    label = torch.tensor([item.tolist() for item in label]).to(torch.int32)
    discourse = torch.tensor([item.tolist() for item in discourse]).to(torch.int32)
    discourse_mask = torch.tensor([item.tolist() for item in discourse_mask]).to(
        torch.int32
    )
    segment_mask = torch.tensor([item.tolist() for item in segment_mask]).to(
        torch.int32
    )
    discourse_adj = torch.tensor([item.tolist() for item in discourse_adj])
    ans_mask = torch.tensor([item.tolist() for item in ans_mask]).to(torch.int32)

    return (
        claim,
        claimId,
        context,
        word_count,
        doc_len,
        clause_len,
        label,
        discourse,
        discourse_mask,
        segment_mask,
        query_len,
        discourse_adj,
        ans_mask,
    )


if __name__ == "__main__":
    configs = Config()
    device = DEVICE
    tokenizer = BertTokenizer.from_pretrained(
        configs.bert_cache_path, cache_dir=CACHE_DIR
    )
    model = Network(configs).to(DEVICE)

    train_dataset = myDataset(tokenizer, configs.train_dataset_path, configs)
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=configs.batch_size,
        collate_fn=my_collate_fn,
    )

    dev_dataset = myDataset(tokenizer, configs.dev_dataset_path, configs)
    dev_loader = DataLoader(
        dataset=dev_dataset, shuffle=True, batch_size=1, collate_fn=my_collate_fn
    )

    test_dataset = myDataset(tokenizer, configs.test_dataset_path, configs)
    test_loader = DataLoader(
        dataset=test_dataset, shuffle=True, batch_size=1, collate_fn=my_collate_fn
    )

    max_micro_result, max_macro_result = main(
        configs, train_loader, dev_loader, test_loader, tokenizer
    )
    # print('max_micro_result: {}, max_macro_result: {}'.format(max_micro_result, max_macro_result))
