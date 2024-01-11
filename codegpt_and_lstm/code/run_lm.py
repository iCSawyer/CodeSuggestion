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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def update_config(args, config):
    config.vocab_size = args.vocab_size


def train(args, train_dataset, model, tokenizer, fh):
    if args.local_rank in [-1, 0]:
        args.tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
        if not os.path.exists(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)
        writer = SummaryWriter(args.tensorboard_dir)

    args.batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size, drop_last=True
    )
    total_examples = len(train_dataset) * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1
    )
    batch_size = (
        args.batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    )

    if args.num_train_epochs > 0:
        t_total = total_examples // batch_size * args.num_train_epochs
    args.max_steps = t_total
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    checkpoint_last = os.path.join(args.output_dir, "checkpoint-last")

    optimizer_last = os.path.join(checkpoint_last, "optimizer.pt")

    if os.path.exists(optimizer_last):
        logger.warning(f"Loading optimizer from {optimizer_last}")
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))
    if args.local_rank == 0:
        torch.distributed.barrier()
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank % args.gpu_per_node],
            output_device=args.local_rank % args.gpu_per_node,
        )

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", total_examples)
    logger.info("  Num epoch = %d", t_total * batch_size // total_examples)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d", batch_size
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb = 0.0, 0.0, 0.0, global_step

    model.zero_grad()
    set_seed(args)

    not_best_time = 0
    not_best_max = 5
    best_acc = -1

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            _idx, _ids = batch["idx"], batch["ids"]
            inputs, labels = (_ids, _ids)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            attn_mask = (inputs != tokenizer.pad_token_id).bool()
            labels[labels == tokenizer.pad_token_id] = -100

            model.train()
            outputs = model(inputs, labels=labels, attention_mask=attn_mask)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if global_step % args.logging_steps == 0:
                    logger.info(
                        "  steps: %s  ppl: %s  lr: %s",
                        global_step,
                        round(avg_loss, 5),
                        scheduler.get_last_lr()[0],
                    )
                    writer.add_scalar("train/ppl", avg_loss, global_step)
                if (
                    args.local_rank in [-1, 0]
                    and args.logging_steps > 0
                    and global_step % args.logging_steps == 0
                ):
                    logging_loss = tr_loss
                    tr_nb = global_step

                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    checkpoint_prefix = "checkpoint"

                    if args.evaluate_during_training:
                        a, b = eval_acc(args, model, tokenizer)
                        cur_acc = b / a
                        output_dir = os.path.join(
                            args.output_dir,
                            "{}-{}-{}".format(checkpoint_prefix, global_step, round(cur_acc, 4)),
                        )

                        if cur_acc > best_acc:
                            best_acc = cur_acc
                            not_best_time = 0
                        else:
                            not_best_time += 1

                    else:
                        output_dir = os.path.join(
                            args.output_dir, "{}-{}".format(checkpoint_prefix, global_step)
                        )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, "module") else model
                    if args.model_type == "rnn":
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
                    else:
                        model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    last_output_dir = os.path.join(args.output_dir, "checkpoint-last")
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    if args.model_type == "rnn":
                        torch.save(
                            model_to_save.state_dict(), os.path.join(last_output_dir, "model.pt")
                        )
                    else:
                        model_to_save.save_pretrained(last_output_dir)
                    tokenizer.save_pretrained(last_output_dir)
                    idx_file = os.path.join(last_output_dir, "idx_file.txt")
                    with open(idx_file, "w", encoding="utf-8") as idxf:
                        idxf.write(str(0) + "\n")

                    torch.save(
                        optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt")
                    )

                    logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, "step_file.txt")
                    with open(step_file, "w", encoding="utf-8") as stepf:
                        stepf.write(str(global_step) + "\n")

            if args.max_steps > 0 and global_step > args.max_steps or not_best_time > not_best_max:
                break
        if args.max_steps > 0 and global_step > args.max_steps or not_best_time > not_best_max:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", eval_when_training=False):
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, file_type="valid")

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True
    )

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in eval_dataloader:
        _idx, _ids = batch["idx"], batch["ids"]
        inputs, labels = (_ids, _ids)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        attn_mask = (inputs != tokenizer.pad_token_id).bool()
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(inputs, labels=labels, attention_mask=attn_mask)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": float(perplexity)}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def eval_acc(args, model, tokenizer, file_type="test", n_exemplars=1):
    eval_dataset = load_and_cache_examples(args, tokenizer, file_type="test")

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
    model.to(args.device)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank % args.gpu_per_node],
            output_device=args.local_rank % args.gpu_per_node,
        )

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
            else:
                total_gt.extend(gt)
                total_pred.extend(pred)

    gt_tokens, pred_tokens = ids2tokens(total_gt, total_pred, tokenizer)

    for x, y in zip(pred_tokens, gt_tokens):
        x = x.replace("\u0120", "")
        y = y.replace("\u0120", "")
        if y not in [SOS, EOS, EOL, PAD, SEP, ""]:
            total += 1
            if x == y:
                correct += 1

    pickle.dump(total_pred, open(os.path.join(args.output_dir, "total_pred.pkl"), "wb"))
    pickle.dump(total_gt, open(os.path.join(args.output_dir, "total_gt.pkl"), "wb"))
    pickle.dump(pred_tokens, open(os.path.join(args.output_dir, "pred_tokens.pkl"), "wb"))
    pickle.dump(gt_tokens, open(os.path.join(args.output_dir, "gt_tokens.pkl"), "wb"))

    saved_file = os.path.join(args.output_dir, "predictions.txt")

    return total, correct


def post_process(args, preds, gts, true_gts, saved_file):
    wf = open(saved_file, "w")

    cnt = 0
    new_gt = []
    new_pred = []
    for i, (pred, gt) in enumerate(zip(preds, gts)):
        if gt in ["", "<pad>"]:
            continue
        new_gt.append(gt)
        new_pred.append(pred.replace(" ", ""))
        if gt == "</s>":
            gt_str = " ".join(new_gt)
            pred_str = " ".join(new_pred)
            assert gt_str == true_gts[cnt].strip(), f"{cnt} sample gt_str != true_gt"
            wf.write(pred_str + "\n")
            cnt += 1
            new_gt = []
            new_pred = []

    return cnt


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

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
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

    parser.add_argument("--tensorboard_dir", type=str)

    args = parser.parse_args()

    args.local_rank = int(os.environ["LOCAL_RANK"])

    args.output_dir = os.path.join(
        args.output_dir,
        args.dataset,
        "with_exemplar_dev" if args.with_exemplar else "without_exemplar_dev",
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
    )
    args.log_file = os.path.join(args.output_dir, "logger.log")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.log_file):
        os.mknod(args.log_file)

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    logger.info(
        "local_rank: %d, node_index: %d, gpu_per_node: %d"
        % (args.local_rank, args.node_index, args.gpu_per_node)
    )

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.local_rank += args.node_index * args.gpu_per_node
        args.n_gpu = 1
    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s, world size: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
        torch.distributed.get_world_size() if args.local_rank != -1 else 1,
    )

    fh = logging.FileHandler(args.log_file)
    logger.addHandler(fh)

    set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.start_epoch = 0
    args.start_step = 0

    checkpoint_last = os.path.join(args.output_dir, "checkpoint-last")
    if args.do_train and os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.pretrain_dir = os.path.join(checkpoint_last)
        args.config_name = os.path.join(checkpoint_last, "config.json")
        idx_file = os.path.join(checkpoint_last, "idx_file.txt")
        with open(idx_file, encoding="utf-8") as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, "step_file.txt")
        if os.path.exists(step_file):
            with open(step_file, encoding="utf-8") as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info(
            "reload model from {}, resume from {} steps".format(checkpoint_last, args.start_step)
        )

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

    if args.local_rank == 0:
        torch.distributed.barrier()

    logger.info("Training/evaluation parameters %s", args)

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, file_type="train")

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, fh)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval:
        test_total, test_cr = eval_acc(args, model, tokenizer, "valid")
        logger.info(f"Test total tokens: {test_total}, accuracy: {test_cr/test_total}")


if __name__ == "__main__":
    main()
