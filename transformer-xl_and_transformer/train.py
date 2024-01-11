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

import os

cpu_num = 4
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)


parser = argparse.ArgumentParser(description="PyTorch Transformer Language Model")
parser.add_argument("--data_dir", type=str, default="../data/", help="location of the data corpus")
parser.add_argument("--dataset", type=str, default="csn", help="dataset name")
parser.add_argument(
    "--with_exemplar", type=lambda x: bool(strtobool(x)), required=True, help="if use exemplar"
)
parser.add_argument(
    "--debug", action="store_true", help="run in debug mode (do not create exp dir)"
)
parser.add_argument("--pretrain", type=str)
parser.add_argument("--lang", type=str)

parser.add_argument("--n_layer", type=int, default=12, help="number of total layers")
parser.add_argument("--n_head", type=int, default=10, help="number of heads")
parser.add_argument("--d_head", type=int, default=50, help="head dimension")
parser.add_argument("--d_embed", type=int, default=-1, help="embedding dimension")
parser.add_argument("--d_model", type=int, default=500, help="model dimension")
parser.add_argument("--d_inner", type=int, default=1000, help="inner dimension in FF")
parser.add_argument("--dropout", type=float, default=0.0, help="global dropout rate")
parser.add_argument("--dropatt", type=float, default=0.0, help="attention probability dropout rate")
parser.add_argument("--init", default="normal", type=str, help="parameter initializer to use.")
parser.add_argument("--emb_init", default="normal", type=str, help="parameter initializer to use.")
parser.add_argument(
    "--init_range",
    type=float,
    default=0.1,
    help="parameters initialized by U(-init_range, init_range)",
)
parser.add_argument(
    "--emb_init_range",
    type=float,
    default=0.01,
    help="parameters initialized by U(-init_range, init_range)",
)
parser.add_argument(
    "--init_std", type=float, default=0.02, help="parameters initialized by N(0, init_std)"
)
parser.add_argument(
    "--proj_init_std", type=float, default=0.01, help="parameters initialized by N(0, init_std)"
)

parser.add_argument(
    "--lr", type=float, default=0.00025, help="initial learning rate (0.00025|5 for adam|sgd)"
)
parser.add_argument("--warmup_step", type=int, default=0, help="upper epoch limit")
parser.add_argument(
    "--decay_rate", type=float, default=0.5, help="decay factor when ReduceLROnPlateau is used"
)
parser.add_argument(
    "--lr_min", type=float, default=0.0, help="minimum learning rate during annealing"
)
parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
parser.add_argument(
    "--clip_nonemb", action="store_true", help="only clip the gradient of non-embedding params"
)
parser.add_argument("--max_step", type=int, default=1000000, help="upper epoch limit")

parser.add_argument("--batch_size", type=int, default=60, help="batch size")
parser.add_argument("--eval_batch_size", type=int, default=10, help="eval batch size")

parser.add_argument("--exemplar_len", type=int, default=256, help="number of tokens to predict")
parser.add_argument("--tgt_len", type=int, default=512, help="number of tokens to predict")
parser.add_argument(
    "--eval_tgt_len", type=int, default=50, help="number of tokens to predict for evaluation"
)
parser.add_argument("--mem_len", type=int, default=0, help="length of the retained previous heads")

parser.add_argument("--seed", type=int, default=1111, help="random seed")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--adaptive", action="store_true", help="use adaptive softmax")
parser.add_argument(
    "--div_val", type=int, default=1, help="divident value for adapative input and softmax"
)
parser.add_argument(
    "--pre_lnorm", action="store_true", help="apply LayerNorm to the input instead of the output"
)
parser.add_argument("--varlen", action="store_true", help="use variable length")
parser.add_argument("--log-interval", type=int, default=200, help="report interval")
parser.add_argument("--eval-interval", type=int, default=4000, help="evaluation interval")
parser.add_argument("--work_dir", default="./save", type=str, help="experiment directory.")

parser.add_argument(
    "--same_length", action="store_true", help="use the same attn length for all tokens"
)
parser.add_argument(
    "--attn_type",
    type=int,
    default=0,
    help="attention type. 0 for ours, 1 for Shaw et al,"
    "2 for Vaswani et al, 3 for Al Rfou et al.",
)
parser.add_argument(
    "--clamp_len", type=int, default=-1, help="use the same pos embeddings after clamp_len"
)
parser.add_argument(
    "--eta_min", type=float, default=0.0, help="min learning rate for cosine scheduler"
)
parser.add_argument("--max_eval_steps", type=int, default=-1, help="max eval steps")
parser.add_argument(
    "--sample_softmax", type=int, default=-1, help="number of samples in sampled softmax"
)
parser.add_argument("--patience", type=int, default=0, help="patience")

args = parser.parse_args()


args.tied = True
if args.d_embed < 0:
    args.d_embed = args.d_model
args.ext_len = 0

args.work_dir = os.path.join(args.work_dir, args.dataset)
if args.with_exemplar:
    work_file_name = "with_exemplar_" + time.strftime("%Y%m%d-%H%M%S")
else:
    work_file_name = "without_exemplar_" + time.strftime("%Y%m%d-%H%M%S")
args.work_dir = os.path.join(args.work_dir, work_file_name)
logging = create_exp_dir(
    args.work_dir, scripts_to_save=["train.py", "mem_transformer.py"], debug=args.debug
)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

tokenizer = get_gpt2_tokenizer(pretrain_file=args.pretrain, lang=args.lang)
vocab_size = len(tokenizer)

cutoffs, tie_projs = [], [False]
if args.adaptive:
    assert args.dataset in ["wt103", "lm1b"]
    if args.dataset == "wt103":
        cutoffs = [20000, 40000, 200000]
        tie_projs += [True] * len(cutoffs)
    elif args.dataset == "lm1b":
        cutoffs = [60000, 100000, 640000]
        tie_projs += [False] * len(cutoffs)


def init_weight(weight):
    if args.init == "uniform":
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == "normal":
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("AdaptiveEmbedding") != -1:
        if hasattr(m, "emb_projs"):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight"):
            init_weight(m.weight)
    elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
        if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, "out_projs"):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("TransformerLM") != -1:
        if hasattr(m, "r_emb"):
            init_weight(m.r_emb)
        if hasattr(m, "r_w_bias"):
            init_weight(m.r_w_bias)
        if hasattr(m, "r_r_bias"):
            init_weight(m.r_r_bias)
        if hasattr(m, "r_bias"):
            init_bias(m.r_bias)


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        if hasattr(m, "p"):
            m.p = args.dropout


def update_dropatt(m):
    if hasattr(m, "dropatt"):
        m.dropatt.p = args.dropatt


model = MemTransformerLM(
    vocab_size,
    args.n_layer,
    args.n_head,
    args.d_model,
    args.d_head,
    args.d_inner,
    args.dropout,
    args.dropatt,
    tie_weight=args.tied,
    d_embed=args.d_embed,
    div_val=args.div_val,
    tie_projs=tie_projs,
    pre_lnorm=args.pre_lnorm,
    tgt_len=args.tgt_len,
    ext_len=args.ext_len,
    mem_len=args.mem_len,
    cutoffs=cutoffs,
    same_length=args.same_length,
    attn_type=args.attn_type,
    clamp_len=args.clamp_len,
    sample_softmax=args.sample_softmax,
)
model.apply(weights_init)
model.word_emb.apply(weights_init)
args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param = sum([p.nelement() for p in model.layers.parameters()])

para_model = model.to(device)

if args.sample_softmax > 0:
    dense_params, sparse_params = [], []
    for param in model.parameters():
        if param.size() == model.word_emb.weight.size():
            sparse_params.append(param)
        else:
            dense_params.append(param)
    optimizer_sparse = optim.SparseAdam(sparse_params, lr=args.lr)
    optimizer = optim.Adam(dense_params, lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_step, eta_min=args.eta_min)
if args.sample_softmax > 0:
    scheduler_sparse = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_sparse, args.max_step, eta_min=args.eta_min
    )

logging("=" * 100)
for k, v in args.__dict__.items():
    logging("    - {} : {}".format(k, v))
logging("=" * 100)
logging("#params = {}".format(args.n_all_param))
logging("#non emb params = {}".format(args.n_nonemb_param))

train_step = 0
train_loss = 0
best_val_loss = 1e9
best_val_acc = -1
not_best_time = 0
not_best_max = 5


def evaluate():
    model.eval()

    if args.mem_len == 0:
        model.reset_length(
            args.eval_tgt_len, args.ext_len + args.tgt_len - args.eval_tgt_len, args.mem_len
        )
    else:
        model.reset_length(
            args.eval_tgt_len, args.ext_len, args.mem_len + args.tgt_len - args.eval_tgt_len
        )

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

    total_len, total_loss = 0, 0.0
    total_token, correct_token = 0, 0
    with torch.no_grad():
        mems = tuple()
        for batch, item in enumerate(eval_dataloader):
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

            if args.max_eval_steps > 0 and batch >= args.max_eval_steps:
                break
            log_probs, ret = model(data, target, data_pad_idx, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.mean()
            total_loss += args.tgt_len * loss.float().item()
            total_len += args.tgt_len

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

            for x, y in zip(pred_tokens, tgt_tokens):
                x = x.replace("\u0120", "")
                y = y.replace("\u0120", "")
                if y not in [SOS, EOS, SEP, PAD, EOL, UNK, ""]:
                    total_token += 1
                    if x == y:
                        correct_token += 1

    model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
    model.train()

    return total_loss / total_len, correct_token / total_token


def train(train_dataloader):
    global train_step, train_loss, best_val_loss, eval_start_time, log_start_time

    total_token, correct_token = 0, 0
    global best_val_acc, best_val_loss, not_best_time, not_best_max

    model.train()
    mems = tuple()
    for batch, item in enumerate(train_dataloader):
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

        model.zero_grad()
        log_probs, ret = para_model(data, target, data_pad_idx, *mems)
        loss, mems = ret[0], ret[1:]
        loss = loss.float().mean().type_as(loss)
        loss.backward()
        train_loss += loss.float().item()

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

        for x, y in zip(pred_tokens, tgt_tokens):
            x = x.replace("\u0120", "")
            y = y.replace("\u0120", "")
            if y not in [SOS, EOS, SEP, PAD, EOL, UNK, ""]:
                total_token += 1
                if x == y:
                    correct_token += 1

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()
        if args.sample_softmax > 0:
            optimizer_sparse.step()

        train_step += 1
        if train_step < args.warmup_step:
            curr_lr = args.lr * train_step / args.warmup_step
            optimizer.param_groups[0]["lr"] = curr_lr
            if args.sample_softmax > 0:
                optimizer_sparse.param_groups[0]["lr"] = curr_lr * 2
        else:
            scheduler.step(train_step)
            if args.sample_softmax > 0:
                scheduler_sparse.step(train_step)

        if train_step % args.log_interval == 0:
            cur_loss = train_loss / args.log_interval
            elapsed = time.time() - log_start_time
            log_str = (
                "| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} "
                "| ms/batch {:5.2f} | loss {:5.2f}".format(
                    epoch,
                    train_step,
                    batch + 1,
                    optimizer.param_groups[0]["lr"],
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                )
            )
            log_str += " | acc {:9.5f}".format(correct_token / total_token)
            logging(log_str)
            train_loss = 0
            correct_token, total_token = 0, 0
            log_start_time = time.time()

        if train_step % args.eval_interval == 0:
            val_loss, val_acc = evaluate()
            logging("-" * 100)
            log_str = "| Eval {:3d} at step {:>8d} | time: {:5.2f}s " "| valid loss {:5.2f}".format(
                train_step // args.eval_interval,
                train_step,
                (time.time() - eval_start_time),
                val_loss,
            )
            log_str += " | valid acc {:9.3f}".format(val_acc)
            logging(log_str)
            logging("-" * 100)
            if not best_val_acc or val_acc > best_val_acc:
                best_val_acc = val_acc
                not_best_time = 0
                if not args.debug:
                    with open(os.path.join(args.work_dir, "model.pt"), "wb") as f:
                        torch.save(model, f)
                    with open(os.path.join(args.work_dir, "optimizer.pt"), "wb") as f:
                        torch.save(optimizer.state_dict(), f)
            else:
                not_best_time += 1
            eval_start_time = time.time()

        if train_step == args.max_step or not_best_time > not_best_max:
            break


log_start_time = time.time()
eval_start_time = time.time()

try:
    train_dataset = exemplarDataset(
        tokenizer,
        args,
        "train",
        args.exemplar_len,
        args.tgt_len,
        n_exemplars=1 if args.with_exemplar else 0,
    )
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True
    )

    for epoch in itertools.count(start=1):
        train(train_dataloader)
        if train_step == args.max_step or not_best_time > not_best_max:
            logging("-" * 100)
            logging("End of training")
            break

except KeyboardInterrupt:
    logging("-" * 100)
    logging("Exiting from training early")

with open(os.path.join(args.work_dir, "model.pt"), "rb") as f:
    model = torch.load(f)
para_model = model.to(device)
