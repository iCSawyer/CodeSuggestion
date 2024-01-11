from __future__ import absolute_import, division, print_function

import argparse
import gc
import glob
import json
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from my_utils import *
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

java_omit = [1986, 2761]
py_omit = [
    9,
    11,
    55,
    56,
    192,
    232,
    251,
    909,
    912,
    913,
    915,
    926,
    932,
    935,
    936,
    941,
    947,
    950,
    951,
    952,
    965,
    967,
    994,
    1002,
    1010,
    1031,
    1032,
    1035,
    1036,
    1039,
    1040,
    1051,
    1087,
    1088,
    1097,
    1103,
    1105,
    1108,
    1112,
    1129,
    1133,
    1135,
    1136,
    1141,
    1498,
    1517,
    1526,
    1602,
    1610,
    1615,
    2115,
    2116,
    2117,
    2118,
    2119,
    2122,
    2126,
    2974,
    2975,
    2981,
    2983,
    3123,
    3787,
    3791,
    3792,
    3807,
    3808,
    3809,
    3820,
    3821,
    3823,
    3824,
    3825,
    3834,
    4104,
    4105,
    4115,
    4159,
    4164,
    4409,
]


class exemplarDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        args,
        logger,
        file_type="valid",
        exemplar_len=200,
        tgt_len=512,
        n_exemplars=1,
    ):
        if args.local_rank == -1:
            local_rank = 0
            world_size = 1
        else:
            local_rank = args.local_rank
            world_size = torch.distributed.get_world_size()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        cache_file = os.path.join(
            args.data_dir,
            args.dataset,
            f"{file_type}_{exemplar_len}_{tgt_len}_{n_exemplars}_codegpt.pkl",
        )
        if os.path.exists(cache_file) and local_rank == 0 and world_size == 1:
            pass

        self.inputs = []
        datafile = os.path.join(args.data_dir, args.dataset, f"{file_type}.jsonl")
        if file_type == "train":
            logger.warning("Creating features from dataset file at %s", datafile)
        with open(datafile) as f:
            data = f.readlines()
        if args.debug:
            data = data[:100]

        logger.info("Data size: %d, Processing ..." % (len(data)))
        for idx, x in enumerate(tqdm(data)):
            js = json.loads(x.strip())

            if args.lang == "java":
                if js["idx"] in java_omit:
                    continue
            elif args.lang == "py":
                if js["idx"] in py_omit:
                    continue
            else:
                raise NotImplementedError

            context = js["context"]
            exemplars = js["exemplars"]

            context = tokenizer.tokenize(context)
            exemplars = [tokenizer.tokenize(x) for x in exemplars if x != ""]

            if n_exemplars != 0 and len(exemplars) == 0:
                sub_tokens = [SOS] + [SEP] + context
                sub_tokens = sub_tokens[: tgt_len - 1] + [EOS]
                sub_tokens = sub_tokens + [PAD] * (tgt_len - len(sub_tokens))
                assert len(sub_tokens) == tgt_len
                ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                if len(ids) == 0:
                    print(ids)
                self.inputs.append({"idx": idx, "ids": ids})
                print("warning")
                continue

            if n_exemplars == 0:
                sub_tokens = [SOS] + context[: tgt_len - 2] + [EOS]
                sub_tokens = sub_tokens + [PAD] * (tgt_len - len(sub_tokens))

                assert len(sub_tokens) == tgt_len
                ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                self.inputs.append({"idx": idx, "ids": ids})
                continue

            if len(exemplars) < n_exemplars:
                exemplars += [exemplars[-1]] * (n_exemplars - len(exemplars))
            exemplars = exemplars[:n_exemplars]

            for exemplar in exemplars:
                exemplar = exemplar[:exemplar_len]

                if tgt_len == 512 and exemplar_len == 255:
                    sub_tokens = [SOS] + exemplar + [SEP] + context[:254]
                elif tgt_len == 256 and exemplar_len == 127:
                    sub_tokens = [SOS] + exemplar + [SEP] + context[:126]
                else:
                    raise NotImplementedError

                sub_tokens = sub_tokens[: tgt_len - 1] + [EOS]
                sub_tokens = sub_tokens + [PAD] * (tgt_len - len(sub_tokens))

                assert len(sub_tokens) == tgt_len
                ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                if len(ids) == 0:
                    print(ids)
                self.inputs.append({"idx": idx, "ids": ids})

        del data
        gc.collect()

        length = len(self.inputs) // world_size
        self.inputs = self.inputs[local_rank * length : (local_rank + 1) * length]

        logger.warning("Local_rank %d, %d samples" % (local_rank, len(self.inputs)))

        if local_rank == 0 and world_size == 1:
            pickle.dump(self.inputs, open(cache_file, "wb"))
            logger.warning("Save to cache: %s" % cache_file)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {"idx": self.inputs[item]["idx"], "ids": torch.tensor(self.inputs[item]["ids"])}


class lineDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        args,
        logger,
        file_type="valid",
        exemplar_len=200,
        tgt_len=512,
        n_exemplars=1,
    ):
        assert file_type == "test" and args.debug == False
        self.inputs = []

        data_file = os.path.join(args.data_dir, args.dataset, f"{file_type}.jsonl")
        with open(data_file) as f:
            lines = f.readlines()
            lines = [json.loads(line.strip()) for line in lines]
        logger.info("Data size: %d, Processing ..." % (len(lines)))

        for idx_line, line in enumerate(tqdm(lines)):
            idx = line["idx"]
            code_tokens = line["context"].split(" ")
            if line["exemplars"] == []:
                exemplar = ""
            else:
                for e in line["exemplars"]:
                    if e != "":
                        exemplar = e
                        break
            if args.lang == "java":
                if idx in java_omit:
                    continue
            elif args.lang == "py":
                if idx in py_omit:
                    continue
            else:
                raise NotImplementedError

            break_pos = []
            if args.lang == "java":
                path = "../../datasets/save/java_breaks.json"
                java_breaks = json.load(open(path))
                break_pos = java_breaks[str(idx)]
            elif args.lang == "py":
                path = "../../datasets/save/py_breaks.json"
                py_breaks = json.load(open(path))
                break_pos = py_breaks[str(idx)]

            for _i in range(len(break_pos) // 2):
                if break_pos[_i + 1] - break_pos[_i] >= 5:
                    break_pos = [break_pos[_i], break_pos[_i + 1]]
                    break
            assert len(break_pos) >= 2

            for i in range(len(break_pos) - 1):
                cxt = " ".join(code_tokens[: break_pos[i] + 1])
                gt = " ".join(code_tokens[break_pos[i] + 1 : break_pos[i + 1] + 1])

                tokenized_cxt = tokenizer.tokenize(cxt)
                tokenized_gt = tokenizer.tokenize(gt)
                if n_exemplars == 0:
                    final_cxt = [SOS] + tokenized_cxt

                    final_cxt = final_cxt[:tgt_len]
                    assert tgt_len == 256
                    final_gt = tokenized_gt
                elif n_exemplars == 1:
                    tokenized_exemplar = tokenizer.tokenize(exemplar)
                    tokenized_exemplar = tokenized_exemplar[:exemplar_len]

                    tokenized_cxt = tokenized_cxt[:255]
                    final_cxt = [SOS] + tokenized_exemplar + [SEP] + tokenized_cxt
                    assert tgt_len == 512, len(final_cxt) <= tgt_len
                    final_gt = tokenized_gt
                self.inputs.append(
                    {
                        "idx": idx,
                        "cxt": tokenizer.convert_tokens_to_ids(final_cxt),
                        "gt": tokenizer.convert_tokens_to_ids(final_gt),
                    }
                )

        del lines
        gc.collect()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {
            "idx": self.inputs[item]["idx"],
            "cxt": torch.tensor(self.inputs[item]["cxt"]),
            "gt": torch.tensor(self.inputs[item]["gt"]),
        }
