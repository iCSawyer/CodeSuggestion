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


def get_start(args, bsz_ids, tokenizer):
    pos = []
    bsz = bsz_ids.shape[0]

    for i in range(bsz):
        ids = bsz_ids[i].cpu().tolist()

        if args.lang == "py":
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

    start_sub_token = bsz_ids[0][pos[0]].item()
    for i in range(bsz):
        assert bsz_ids[i][pos[i]].item() == start_sub_token

    assert len(pos) == bsz

    return start_sub_token, pos


def eval_acc(args, model, tokenizer, file_type="test"):
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

    eval_dataset = load_and_cache_examples(args, tokenizer, file_type="test")
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
    model.to(args.device)

    model.eval()

    edit_sim = 0.0
    em = 0.0
    bleu1, bleu2, bleu3, bleu4 = 0.0, 0.0, 0.0, 0.0
    N = 0

    for step, batch in enumerate(tqdm(eval_dataloader)):
        _idx, _ids = batch["idx"], batch["ids"]

        bsz = _ids.size(0)
        assert bsz == 1 if not args.with_exemplar else bsz == args.n_exemplars

        _sos, split_pos = get_start(args, _ids, tokenizer)
        gt = _ids[0][split_pos[0] + 1 :].tolist()

        past_key_values_dic = {}
        for i in range(bsz):
            _input = _ids[i][: split_pos[i]].to(args.device)
            if _input.size(0) == 0:
                past_key_values_dic[i] = None
            else:
                _mask = (_input != tokenizer.pad_token_id).bool()
                _outputs = model(_input, attention_mask=_mask, use_cache=True)
                past_key_values_dic[i] = _outputs[1]

        m = torch.nn.LogSoftmax(dim=-1)
        cur_input_ids = torch.tensor(_sos).unsqueeze(0).to(args.device)
        for offset in range(100):
            all_pred_scores = []
            for i in range(bsz):
                _input = _ids[i][: split_pos[i]].to(args.device)
                _input = torch.cat((_input, cur_input_ids))
                _input = _input.unsqueeze(0)
                _mask = (_input != tokenizer.pad_token_id).bool()

                _outputs = model(
                    cur_input_ids[-1:].unsqueeze(0),
                    attention_mask=_mask,
                    use_cache=True,
                    past_key_values=past_key_values_dic[i],
                )

                _score = _outputs[0][:, -1, :]
                past_key_values_dic[i] = _outputs[1]
                all_pred_scores.append(_score)

            all_pred_scores = torch.stack(all_pred_scores)
            all_pred_scores = m(all_pred_scores)

            all_pred_scores = all_pred_scores.mean(dim=0)
            next_id = all_pred_scores.argmax(-1)

            next_id = all_pred_scores.argmax(-1)

            stop_list = [EOS] if args.lang == "py" else ["Ġ;", EOS]
            if tokenizer.convert_ids_to_tokens(next_id)[0] in stop_list:
                break
            cur_input_ids = torch.cat((cur_input_ids, next_id), dim=0)

        pred_ids = cur_input_ids[1:].cpu().tolist()
        tgt_ids = _ids[0][split_pos[0] + 1 :].cpu().tolist()

        pred = DecodeIds(pred_ids).strip()
        tgt = DecodeIds(tgt_ids).strip()

        if args.lang == "py":
            end_idx = tgt.index(EOS)
        else:
            try:
                end_idx = tgt.index(";")
            except:
                end_idx = tgt.index(EOS)

        tgt = tgt[:end_idx].strip()
        pred = pred.strip()

        def f(code):
            code = " ".join(code.split())
            code = code.replace("<EOL>", "")

            if re.search(r"<(NUM|STR|CHAR)_LIT>", code):
                code = code.replace("<NUM_LIT>", "0")
                code = code.replace("<STR_LIT>", "str")
                code = code.replace("<CHAR_LIT>", "c")
                code = re.sub(r"<NUM_LIT:(\d+)>", r"\1", code)
                code = re.sub(r"<STR_LIT:(.*?)>", r"\1", code)
                code = re.sub(r"<CHAR_LIT:(.*?)>", r"\1", code)
            code = re.sub(r"([^A-Za-z0-9_])", r" \1 ", code)
            code = re.sub(r"([a-z])([A-Z])", r"\1 \2", code)
            code = re.sub(r"\s+", " ", code)
            code = code.replace('"', "`")
            code = code.replace("'", "`")
            return code.split()

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
        if step % 100 == 0:
            logger.info(
                f"ES:{edit_sim/N:.4f}, EM:{em/N:.4f}, BLEU1:{bleu1/N:.4f}, BLEU2:{bleu2/N:.4f}, BLEU3:{bleu3/N:.4f}, BLEU4:{bleu4/N:.4f}"
            )

    logger.info(
        f"ES:{edit_sim/N:.4f}, EM:{em/N:.4f}, BLEU1:{bleu1/N:.4f}, BLEU2:{bleu2/N:.4f}, BLEU3:{bleu3/N:.4f}, BLEU4:{bleu4/N:.4f}"
    )


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
    parser.add_argument("--n_exemplars", type=int, default=-1, help="num of gpus per node")

    args = parser.parse_args()

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
            eval_acc(args, model, tokenizer, "valid")


if __name__ == "__main__":
    main()
