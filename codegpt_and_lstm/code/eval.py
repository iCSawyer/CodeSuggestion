"""
Code completion (both token level and line level) pipeline in CodeXGLUE
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import os
import pdb
import pickle
import random
import re
import shutil
import time
import warnings
from distutils.util import strtobool

import numpy as np
import torch
import torch.profiler
from tensorboardX import SummaryWriter
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
    GPT2TokenizerFast,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from beam import Beam
from dataset import exemplarDataset
from model import RNNModel
from my_utils import *
import debugpy
from tokenize import tokenize, NAME, OP
from io import BytesIO
import javalang


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast),
    "rnn": (GPT2Config, RNNModel, GPT2TokenizerFast),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
}


def load_and_cache_examples(args, tokenizer, file_type="valid"):
    if args.with_exemplar:
        if file_type in ["train"]:
            n_exemplars = 1
        elif file_type in ["valid", "test"]:
            n_exemplars = 1
    else:
        n_exemplars = 0

    logger.warning(f"Loading {file_type} examples with {n_exemplars} exemplars")

    dataset = exemplarDataset(
        tokenizer,
        args,
        logger,
        file_type=file_type,
        exemplar_len=args.exemplar_len,
        tgt_len=args.tgt_len,
        n_exemplars=n_exemplars,
    )
    return dataset


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


def eval_acc(args, model, tokenizer, file_type="test", n_exemplars=1):
    _tokens = []
    result = {}
    assert args.local_rank in [-1, 0]

    eval_dataset = load_and_cache_examples(args, tokenizer, file_type="test")
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
    model.to(args.device)

    model.eval()

    correct = 0.0
    total = 0

    total_pred = []
    total_gt = []

    for step, batch in enumerate(tqdm(eval_dataloader)):
        _idx, _ids = batch["idx"], batch["ids"]
        inputs = _ids.to(args.device)

        attn_mask = (inputs != tokenizer.pad_token_id).bool()

        with torch.no_grad():
            outputs = model(inputs, attention_mask=attn_mask)
            pred_scores = outputs[0]
            pred_ids = pred_scores.argmax(-1)

        for pred, gt in zip(pred_ids, inputs):
            pred = pred.cpu().tolist()[:-1]
            gt = gt.cpu().tolist()[1:]

            if args.with_exemplar:
                idx = gt.index(tokenizer.convert_tokens_to_ids(SEP))
                total_gt.extend(gt[idx:])
                total_pred.extend(pred[idx:])
                _gt, _pred = ids2tokens(gt[idx:], pred[idx:], tokenizer)
            else:
                total_gt.extend(gt)
                total_pred.extend(pred)
                _gt, _pred = ids2tokens(gt, pred, tokenizer)

        _tokens.append((_pred, _gt))

    gt_tokens, pred_tokens = ids2tokens(total_gt, total_pred, tokenizer)

    for x, y in zip(pred_tokens, gt_tokens):
        x = x.replace("\u0120", "")
        y = y.replace("\u0120", "")
        if y not in [SOS, EOS, SEP, PAD, EOL, UNK, ""]:
            total += 1
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
                correct += 1
                result[typ] = (result[typ][0] + 1, result[typ][1] + 1)
            else:
                result[typ] = (result[typ][0], result[typ][1] + 1)

    for key in result:
        logging.info(
            f"Type:{key:12s}, Part:{result[key][1]/total:6.5f}, Acc:{result[key][0]/result[key][1]:6.5f}"
        )
    logging.info(f"Type:Overall, Part:{total/total:6.5f}, Acc:{correct/total:6.5f}")

    with open("_a.json", "w") as f:
        json.dump(_tokens, f)

    return total, correct


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tgt_len", default=512, type=int, required=True)
    parser.add_argument("--exemplar_len", default=200, type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True, help="get special tokens")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--data_dir", default="../../datasets/", type=str, help="e.g. data_dir/dataset/train.jsonl"
    )
    parser.add_argument(
        "--output_dir", default="../save", type=str, help="output dir"
    )
    parser.add_argument("--with_exemplar", type=lambda x: bool(strtobool(x)), required=True)

    parser.add_argument(
        "--model_type", default="gpt2", type=str, help="The model architecture to be fine-tuned."
    )
    parser.add_argument(
        "--pretrain_dir",
        default="",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--config_dir", type=str, help="config name. Required when training from scratch"
    )
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        help="Pre-trained tokenizer dir. Required when training from scratch",
    )
    parser.add_argument(
        "--load_name", type=str, default="pretrained", help="Load pretrained model name"
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)",
    )

    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=12,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=1000, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps."
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--node_index", type=int, default=-1, help="node index if multi-node running"
    )
    parser.add_argument("--gpu_per_node", type=int, default=-1, help="num of gpus per node")

    args = parser.parse_args()

    args.local_rank = -1
    args.n_gpu = 1

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    fh = logging.FileHandler("./eval.log")
    logger.addHandler(fh)

    special_tokens = get_special_tokens(lang=args.lang)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    pretrained = args.pretrain_dir
    if pretrained:
        tokenizer = tokenizer_class.from_pretrained(
            pretrained,
            do_lower_case=args.do_lower_case,
            sep_token="<EOL>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<|UNKNOWN|>",
            additional_special_tokens=special_tokens,
        )
        if args.model_type == "rnn":
            model = model_class(len(tokenizer), 768, 768, 1)
            model_last = os.path.join(pretrained, "model.pt")
            if os.path.exists(model_last):
                logger.warning(f"Loading model from {model_last}")
                model.load_state_dict(torch.load(model_last, map_location="cpu"))
        else:
            model = model_class.from_pretrained(pretrained)
            model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_dir,
            sep_token="<EOL>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<|UNKNOWN|>",
            additional_special_tokens=special_tokens,
        )
        args.vocab_size = len(tokenizer)
        if args.model_type == "rnn":
            model = model_class(len(tokenizer), 768, 768, 1)
        else:
            config = config_class.from_pretrained(args.config_dir)
            model = model_class(config)
            model.resize_token_embeddings(len(tokenizer))

    model_parameters = model.parameters()
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f"Model has a total of {num_params} trainable parameters")

    logger.info("Training/evaluation parameters %s", args)

    if args.do_eval:
        eval_acc(args, model, tokenizer, "valid")


if __name__ == "__main__":
    main()
