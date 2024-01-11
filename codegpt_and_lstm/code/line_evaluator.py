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
from dataset import exemplarDataset, lineDataset
from model import RNNModel
from my_utils import *
import debugpy
from tokenize import tokenize, NAME, OP
from io import BytesIO
import javalang
from fuzzywuzzy import fuzz
from tqdm import trange
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast),
    "rnn": (GPT2Config, RNNModel, GPT2TokenizerFast),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
}


def load_and_cache_examples(args, tokenizer, file_type="valid"):
    n_exemplars = args.n_exemplars
    assert n_exemplars <= 1
    logger.warning(f"Loading {file_type} examples with {n_exemplars} exemplars")

    dataset = lineDataset(
        tokenizer,
        args,
        logger,
        file_type=file_type,
        exemplar_len=args.exemplar_len,
        tgt_len=args.tgt_len,
        n_exemplars=n_exemplars,
    )
    return dataset


def eval_line(args, model, tokenizer):
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

    eval_dataset = load_and_cache_examples(args, tokenizer, file_type="test")
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    assert args.eval_batch_size == 1
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
    model.to(args.device)
    model.eval()
    sf = SmoothingFunction().method2

    edit_sim = 0.0
    em = 0.0
    bleu1, bleu2, bleu3, bleu4 = 0.0, 0.0, 0.0, 0.0
    N = 0
    for step, batch in enumerate(tqdm(eval_dataloader)):
        idx, cxt, gt = batch["idx"], batch["cxt"], batch["gt"]
        cxt = cxt.to(args.device)

        output = model.generate(
            cxt,
            max_new_tokens=len(gt[0]),
            num_return_sequences=1,
            early_stopping=True,
        )

        output = tokenizer.decode(output[0])[len(tokenizer.decode(cxt[0])) :]
        cxt = tokenizer.decode(cxt[0])
        gt = tokenizer.decode(gt[0])

        output = output.replace("<sep>", "").replace("<s>", "").replace("</s>", "").strip()
        gt = gt.strip()

        p = f(output)
        g = f(gt)

        if p == g:
            em += 1
        edit_sim += fuzz.ratio(" ".join(p), " ".join(g))

        bleu1 += bleu([g], p, weights=(1, 0, 0, 0), smoothing_function=sf)
        bleu2 += bleu([g], p, weights=(0.5, 0.5, 0, 0), smoothing_function=sf)
        bleu3 += bleu([g], p, weights=(0.33, 0.33, 0.33, 0), smoothing_function=sf)
        bleu4 += bleu([g], p, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf)

        N += 1

    logger.info(
        f"ES:{edit_sim/N:.4f}, EM:{em/N:.4f}, BLEU1:{bleu1/N:.4f}, BLEU2:{bleu2/N:.4f}, BLEU3:{bleu3/N:.4f}, BLEU4:{bleu4/N:.4f}"
    )


def set_args():
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
    parser.add_argument("--n_exemplars", type=int, default=-1, help="num of gpus per node")

    args = parser.parse_args()
    return args


def main():
    args = set_args()

    if args.debug:
        print("Waiting for debugger attach")
        debugpy.listen(("localhost", 7451))
        debugpy.wait_for_client()

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
        with torch.no_grad():
            eval_line(args, model, tokenizer)


if __name__ == "__main__":
    main()
