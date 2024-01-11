import argparse
import itertools
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from data_utils import exemplarDataset
from mem_transformer import MemTransformerLM
from my_utils import *
from utils.data_parallel import BalancedDataParallel
from utils.exp_utils import create_exp_dir
import debugpy
from distutils.util import strtobool
from tqdm import tqdm, trange
import gc
import inspect
from tokenize import tokenize, NAME, OP
from io import BytesIO
import javalang

parser = argparse.ArgumentParser(description="PyTorch Transformer Language Model")


parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--dataset", type=str, default="csn", help="dataset name")
parser.add_argument(
    "--with_exemplar", type=lambda x: bool(strtobool(x)), required=True, help="if use exemplar"
)
parser.add_argument(
    "--attn_type",
    type=int,
    default=0,
    help="attention type. 0 for ours, 1 for Shaw et al,"
    "2 for Vaswani et al, 3 for Al Rfou et al.",
)
parser.add_argument("--exemplar_len", type=int, default=256, help="number of tokens to predict")
parser.add_argument("--tgt_len", type=int, default=512, help="number of tokens to predict")
parser.add_argument(
    "--eval_tgt_len", type=int, default=50, help="number of tokens to predict for evaluation"
)
parser.add_argument("--mem_len", type=int, default=0, help="length of the retained previous heads")


parser.add_argument(
    "--debug", action="store_true", help="run in debug mode (do not create exp dir)"
)
parser.add_argument("--eval_batch_size", type=int, default=10, help="eval batch size")
parser.add_argument("--data_dir", type=str, default="../data/", help="location of the data corpus")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--work_dir", default="./save", type=str, help="experiment directory.")
parser.add_argument("--pretrain", type=str)
parser.add_argument("--lang", type=str)


parser.add_argument("--max_eval_steps", type=int, default=-1, help="max eval steps")
parser.add_argument("--batch_size", type=int, default=-1, help="batch size")

args = parser.parse_args()
if args.debug:
    print("Waiting for debugger attach")
    debugpy.listen(("localhost", 7451))
    debugpy.wait_for_client()

args.ext_len = 0

device = torch.device("cuda" if args.cuda else "cpu")

tokenizer = get_gpt2_tokenizer(pretrain_file=args.pretrain, lang=args.lang)

with open(os.path.join(args.model_dir, "model.pt"), "rb") as f:
    model = torch.load(f)

model.backward_compatible()
model = model.to(device)


model.reset_length(args.eval_tgt_len, args.ext_len, args.mem_len)


def judge(lang, s):
    if lang == "java":
        try:
            typ = type(list(javalang.tokenizer.tokenize(s))[0])
        except:
            typ = "Unknown"

        if "Identifier" in str(typ):
            typ = "Identifier"
        elif "Separator" in str(typ):
            typ = "Separator"
        elif "Keyword" in str(typ):
            typ = "Keyword"
        elif "Operator" in str(typ):
            typ = "Operator"
        else:
            typ = "Others"
    else:
        ops = [
            r"+",
            r"-",
            r"*",
            r"**",
            r"/",
            r"//",
            r"%",
            r"@",
            r"<<",
            r">>",
            r"&",
            r"|",
            r"^",
            r"~",
            r":=",
            r"<",
            r">",
            r"<=",
            r">=",
            r"==",
            r"!=",
        ]
        seps = [
            r"(",
            r")",
            r"[",
            r"]",
            r"{",
            r"}",
            r",",
            r":",
            r".",
            r";",
            r"@",
            r"=",
            r"->",
            r"+=",
            r"-=",
            r"*=",
            r"/=",
            r"//=",
            r"%=",
            r"@=",
            r"&=",
            r"|=",
            r"^=",
            r">>=",
            r"<<=",
            r"**=",
        ]
        keys = [
            r"False",
            r"None",
            r"True",
            r"and",
            r"as",
            r"assert",
            r"async",
            r"await",
            r"break",
            r"class",
            r"continue",
            r"def",
            r"del",
            r"elif",
            r"else",
            r"except",
            r"finally",
            r"for",
            r"from",
            r"global",
            r"if",
            r"import",
            r"in",
            r"is",
            r"lambda",
            r"nonlocal",
            r"not",
            r"or",
            r"pass",
            r"raise",
            r"return",
            r"try",
            r"while",
            r"with",
            r"yield",
        ]
        try:
            g = tokenize(BytesIO(s.encode("utf-8")).readline)
            tp, val, _, _, _ = list(g)[1]
            if tp == NAME:
                if val in keys:
                    typ = "Keyword"
                else:
                    typ = "Identifier"
            elif tp == OP:
                if val in ops:
                    typ = "Operator"
                elif val in seps:
                    typ = "Separator"
                else:
                    typ = "Others"
            else:
                typ = "Others"
        except:
            print(f"Error: {s}, {tp}")
            typ = "Others"
    return typ


log_start_time = time.time()
eval_start_time = time.time()
result = {}


def evaluate():
    model.eval()

    eval_dataset = exemplarDataset(
        tokenizer,
        args,
        "test",
        args.exemplar_len,
        args.eval_tgt_len,
        n_exemplars=1 if args.with_exemplar else 0,
    )
    eval_sampler = SequentialSampler(eval_dataset)
    assert args.eval_batch_size == 1
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=False
    )

    total_token, correct_token = 0, 0
    mems = tuple()

    with torch.no_grad():
        for batch, item in enumerate(tqdm(eval_dataloader)):
            ids = item["ids"].to(device)
            data = ids.t().contiguous()[:-1, :]
            target = ids.t().contiguous()[1:, :]

            data_pad_idx = []
            bsz = data.size(1)
            for i in range(bsz):
                cur_seq = data[:, i].cpu().tolist()
                try:
                    cur_pad_idx = cur_seq.index(tokenizer.pad_token_id)
                except ValueError:
                    cur_pad_idx = len(cur_seq)
                data_pad_idx.append(cur_pad_idx)
            data_pad_idx = torch.tensor(data_pad_idx, device=device)

            log_probs, ret = model(data, target, data_pad_idx, *mems)
            _, mems = ret[0], ret[1:]

            tgt_ids = []
            real_pad_idx = []
            for i in range(bsz):
                try:
                    p_i = target[:, i].tolist().index(tokenizer.pad_token_id)
                    cur_ids = target[:p_i, i].tolist()
                    real_pad_idx.append(p_i)
                except:
                    cur_ids = target[:, i].tolist()
                    real_pad_idx.append(len(cur_ids))
                tgt_ids.append(cur_ids)
            data_ids = [data[: data_pad_idx[i], i].tolist() for i in range(bsz)]

            total_pred_ids = log_probs.argmax(-1).tolist()
            pred_ids = [[] for _ in range(bsz)]
            i = 0
            while i < len(total_pred_ids):
                flag = 0
                for j in range(bsz):
                    if len(pred_ids[j]) < real_pad_idx[j]:
                        pred_ids[j].extend([total_pred_ids[i]])
                        i += 1
                        flag = 1
                if flag == 0:
                    break

            flatten_tgt_ids = []
            flatten_pred_ids = []
            SEP_id = tokenizer.convert_tokens_to_ids(SEP)
            for tgt_sublist, pred_sublist in zip(tgt_ids, pred_ids):
                if args.with_exemplar:
                    SEP_idx = tgt_sublist.index(SEP_id)
                    flatten_tgt_ids.extend(tgt_sublist[SEP_idx:])
                    flatten_pred_ids.extend(pred_sublist[SEP_idx:])
                else:
                    flatten_tgt_ids.extend(tgt_sublist[:])
                    flatten_pred_ids.extend(pred_sublist[:])
            tgt_tokens, pred_tokens = ids2tokens(flatten_tgt_ids, flatten_pred_ids, tokenizer)
            print(tgt_tokens)

            for x, y in zip(pred_tokens, tgt_tokens):
                x = x.replace("\u0120", "")
                y = y.replace("\u0120", "")
                if y not in [SOS, EOS, SEP, PAD, EOL, UNK, ""]:
                    total_token += 1
                    try:
                        typ = judge(args.lang, y)
                    except:
                        typ = "Unknown"

                    if "Identifier" in str(typ):
                        typ = "Identifier"
                    elif "Separator" in str(typ):
                        typ = "Separator"
                    elif "Keyword" in str(typ):
                        typ = "Keyword"
                    elif "Operator" in str(typ):
                        typ = "Operator"
                    else:
                        typ = "Others"

                    result[typ] = result.get(typ, (0, 0))
                    if x == y:
                        correct_token += 1
                        result[typ] = (result[typ][0] + 1, result[typ][1] + 1)
                    else:
                        result[typ] = (result[typ][0], result[typ][1] + 1)

    for key in result:
        print(
            f"Type:{key:12s}, Part:{result[key][1]/total_token:6.5f}, Acc:{result[key][0]/result[key][1]:6.5f}"
        )
    print(
        f"Type:Overall, Part:{total_token/total_token:6.5f}, Acc:{correct_token/total_token:6.5f}"
    )

    return correct_token / total_token


evaluate()
