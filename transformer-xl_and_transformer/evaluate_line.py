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
from fuzzywuzzy import fuzz
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction

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
parser.add_argument("--n_exemplars", type=int)

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


def get_start(lang, bsz_ids, tokenizer):
    pos = []
    bsz = bsz_ids.shape[1]

    for i in range(bsz):
        ids = bsz_ids[:, i].cpu().tolist()
        if lang == "py":
            sub_tokens = tokenizer.convert_ids_to_tokens(ids)
            N = 7
            for j in range(len(sub_tokens) - 1, -1, -1):
                if sub_tokens[j] in [PAD, EOS]:
                    continue
                if sub_tokens[j] in [SEP, SOS]:
                    pos.append(j)
                    break
                if sub_tokens[j].startswith(("Ġ", "<NUM_LIT")):
                    N -= 1
                    if N == 0:
                        pos.append(j - 1)
                        break
        else:
            _f = False
            for j in range(len(ids) - 1, -1, -1):
                if ids[j] in tokenizer.convert_tokens_to_ids([SOS, SEP, "Ġ{"]):
                    pos.append(j)
                    break

                if ids[j] in tokenizer.convert_tokens_to_ids(["Ġ;"]):
                    if _f:
                        pos.append(j)
                        break
                    else:
                        _f = True

    start_sub_token = bsz_ids[:, 0][pos[0]].item()
    for i in range(bsz):
        assert bsz_ids[:, i][pos[i]].item() == start_sub_token

    assert len(pos) == bsz

    return start_sub_token, pos


def get_start_pos_and_len(args, idx, tokenizer, dataset: str, cur_ids):
    idx = idx.item()

    path = f"datasets/{dataset}/test.jsonl"
    with open(path) as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    dataset_entry = None
    for entry in lines:
        if str(entry["idx"]) == str(idx):
            dataset_entry = entry
            break

    path = f"../datasets/save/{args.lang}_breaks.json"
    breaks = json.load(open(path))
    break_pos = breaks[str(idx)]

    for _i in range(len(break_pos) // 2):
        if break_pos[_i + 1] - break_pos[_i] >= 5:
            break_pos = [break_pos[_i], break_pos[_i + 1]]
            break
    assert len(break_pos) >= 2

    s1 = " ".join(dataset_entry["context"].split(" ")[: break_pos[0] + 1])
    s2 = " ".join(dataset_entry["context"].split(" ")[: break_pos[1] + 1])
    for x in dataset_entry["exemplars"]:
        if len(x) > 0:
            s0 = x
            break

    n_s1 = len(tokenizer.tokenize(s1))
    n_s2 = len(tokenizer.tokenize(s2))
    try:
        n_s0 = len(tokenizer.tokenize(s0)[:255])
    except:
        n_s0 = 0

    if args.with_exemplar:
        assert tokenizer.tokenize(s1)[-1] == tokenizer.convert_ids_to_tokens(
            cur_ids[n_s1 + n_s0 + 1].item() if n_s0 > 0 else cur_ids[n_s1 + 1].item()
        )

    else:
        assert tokenizer.tokenize(s1)[-1] == tokenizer.convert_ids_to_tokens(cur_ids[n_s1].item())

    return n_s0, n_s1, n_s2 - n_s1


log_start_time = time.time()
eval_start_time = time.time()
result = {}


def evaluate():
    def DecodeIds(idxs):
        codes = ""
        for idx in idxs:
            to_add = tokenizer.convert_ids_to_tokens(idx)
            if tokenizer.convert_ids_to_tokens(idx)[0] == "\u0120":
                if not codes.endswith(" "):
                    codes += " " + to_add[1:]
                else:
                    codes += to_add[1:]
            elif idx in [
                tokenizer.bos_token_id,
                tokenizer.eos_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
            ] or tokenizer.convert_ids_to_tokens(idx).startswith("<NUM_LIT"):
                codes += " " + to_add + " "
            else:
                codes += to_add
        return codes.strip(" ")

    def f(code):
        code = " ".join(code.split())
        code = code.replace("<EOL>", "")

        code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "str").replace("<CHAR_LIT>", "c")
        pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
        lits = re.findall(pattern, code)
        for lit in lits:
            code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])

        code = re.sub(r"([^A-Za-z0-9_])", r" \1 ", code)
        code = re.sub(r"([a-z])([A-Z])", r"\1 \2", code)
        code = re.sub(r"\s+", " ", code)
        code = code.replace('"', "`")
        code = code.replace("'", "`")
        return code.split()

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
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True
    )

    edit_sim, em, bleu1, bleu2, bleu3, bleu4 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    N = 0
    mems = tuple()
    with torch.no_grad():
        bar = tqdm(eval_dataloader, desc=f"ES:{edit_sim}, EM:{em}, BLEU4:{bleu4}")
        for batch, item in enumerate(bar):
            idx = item["idx"]
            ids = item["ids"].to(device)
            data = ids.t().contiguous()[:-1, :].to(device)
            target = ids.t().contiguous()[1:, :].to(device)
            bsz = data.size(1)

            data_pad_idx = []
            for i in range(bsz):
                cur_seq = data[:, i].cpu().tolist()
                try:
                    cur_pad_idx = cur_seq.index(tokenizer.pad_token_id)
                except ValueError:
                    cur_pad_idx = len(cur_seq)
                data_pad_idx.append(cur_pad_idx)
            data_pad_idx = torch.tensor(data_pad_idx, device=device)

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

            assert bsz == 1 if not args.with_exemplar else True

            exem_s0, cxt_len, gt_len = get_start_pos_and_len(
                args, idx, tokenizer, args.dataset, data[:, 0]
            )
            if args.with_exemplar:
                split_pos = [1 + exem_s0 + 1 + cxt_len - 1]
                _sos = data[1 + exem_s0 + 1 + cxt_len - 1, 0]
            else:
                split_pos = [1 + cxt_len - 1]
                _sos = data[1 + cxt_len - 1, 0]

            cur_input_ids = torch.tensor(_sos).unsqueeze(0).to(device)
            for offset in range(gt_len):
                _input = data[: split_pos[i], i].to(device)
                _input = torch.cat((_input, cur_input_ids))
                _input = _input.unsqueeze(0).t()
                _log_probs, ret = model(
                    _input,
                    target[: split_pos[i] + offset + 1, i].unsqueeze(0).t(),
                    data_pad_idx,
                    *mems,
                )

                all_pred_scores = _log_probs[-1:, :]
                next_id = all_pred_scores.argmax(-1)
                cur_input_ids = torch.cat((cur_input_ids, next_id), dim=0)

            pred_ids = cur_input_ids[1:].cpu().tolist()
            tgt_ids = target[split_pos[0] :, 0].cpu().tolist()

            pred = DecodeIds(pred_ids).strip()
            tgt = DecodeIds(tgt_ids[:gt_len]).strip()

            pred = pred.replace("<sep>", "").replace("<s>", "").replace("</s>", "").strip()
            tgt = tgt.strip()

            p = f(pred)
            g = f(tgt)

            if g == p:
                em += 1
            edit_sim += fuzz.ratio(" ".join(p), " ".join(g))
            sf = SmoothingFunction().method2
            bleu1 += bleu([g], p, weights=(1, 0, 0, 0), smoothing_function=sf)
            bleu2 += bleu([g], p, weights=(0.5, 0.5, 0, 0), smoothing_function=sf)
            bleu3 += bleu([g], p, weights=(0.33, 0.33, 0.33, 0), smoothing_function=sf)
            bleu4 += bleu([g], p, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf)

            N += 1

            _, ret = model(data[:, -1:], target[:, -1:], data_pad_idx[-1:], *mems)
            _, mems = ret[0], ret[1:]

            bar.set_description(f"ES:{edit_sim/N:.2f}, EM:{em/N*100:.2f}, BLEU4:{bleu4/N*100:.2f}")

        print(
            f"ES:{edit_sim/N:.4f}, EM:{em/N:.4f}, BLEU1:{bleu1/N:.4f}, BLEU2:{bleu2/N:.4f}, BLEU3:{bleu3/N:.4f}, BLEU4:{bleu4/N:.4f}"
        )


evaluate()
