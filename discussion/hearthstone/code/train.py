from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import random
import re
import time
import warnings
from distutils.util import strtobool

import debugpy
import numpy as np
import torch
from dataset import trainDataset, testDataset
from my_utils import *
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import (
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast),
}


def process_for_eval(code):
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
    tokens = [t for t in code.split(" ") if t]

    return " ".join(tokens)


def load_and_cache_examples(args, tokenizer, file_type="valid"):
    if args.with_exemplar:
        n_exemplars = 1
    else:
        n_exemplars = 0

    if file_type == "train":
        dataset = trainDataset(
            tokenizer,
            args,
            logger,
            file_type=file_type,
            exemplar_len=args.exemplar_len,
            tgt_len=args.tgt_len,
            n_exemplars=n_exemplars,
        )
        e = dataset[0]["ids"]
        logging.info(tokenizer.decode(e))
    elif file_type == "test":
        dataset = testDataset(
            tokenizer,
            args,
            logger,
            file_type=file_type,
            exemplar_len=args.exemplar_len,
            tgt_len=args.tgt_len,
            n_exemplars=n_exemplars,
        )
        e = dataset[0]["in"]
        ee = dataset[0]["out"]
        logging.info(tokenizer.decode(e))
        logging.info(tokenizer.decode(ee))

    logger.warning(f"Loading {file_type} examples with {n_exemplars} exemplars")

    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
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

    assert not args.fp16

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
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        batch_size,
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

            model.train()
            attn_mask = (inputs != tokenizer.pad_token_id).bool()
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(inputs, labels=labels, attention_mask=attn_mask)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            assert not args.fp16
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
                        me = eval_metric(args, model, tokenizer)

                        cur_acc = me

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
                            args.output_dir,
                            "{}-{}".format(checkpoint_prefix, global_step),
                        )

                    if best_acc == cur_acc:
                        last_output_dir = os.path.join(args.output_dir, "checkpoint-best")
                        if not os.path.exists(last_output_dir):
                            os.makedirs(last_output_dir)
                        model_to_save = model.module if hasattr(model, "module") else model
                        torch.save(
                            model_to_save.state_dict(),
                            os.path.join(last_output_dir, "model.pt"),
                        )
                        model_to_save.save_pretrained(last_output_dir)
                        tokenizer.save_pretrained(last_output_dir)
                        idx_file = os.path.join(last_output_dir, "idx_file.txt")
                        with open(idx_file, "w", encoding="utf-8") as idxf:
                            idxf.write(str(0) + "\n")
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(last_output_dir, "optimizer.pt"),
                        )
                        logger.info(
                            "Saving optimizer and scheduler states to %s",
                            last_output_dir,
                        )
                        step_file = os.path.join(last_output_dir, "step_file.txt")
                        with open(step_file, "w", encoding="utf-8") as stepf:
                            stepf.write(str(global_step) + "\n")

            if args.max_steps > 0 and global_step > args.max_steps or not_best_time > not_best_max:
                break
        if args.max_steps > 0 and global_step > args.max_steps or not_best_time > not_best_max:
            break

    return global_step, tr_loss / global_step


def eval_metric(args, model, tokenizer):
    eval_dataset = load_and_cache_examples(args, tokenizer, file_type="test")
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
    assert args.n_gpu == 1
    model.to(args.device)
    model.eval()

    datafile = os.path.join(args.data_dir, args.dataset, "test.jsonl")
    sf = SmoothingFunction().method2
    with open(datafile, "r") as f:
        lines = f.readlines()
        real_gt = [process_for_eval(json.loads(line.strip())["code"]) for line in lines]
    assert len(real_gt) == len(eval_dataset)

    N = 0
    em, bleu4 = 0.0, 0.0
    bar = tqdm(eval_dataloader)
    for step, batch in enumerate(bar):
        _idx, _in, _out = batch["idx"], batch["in"], batch["out"]
        assert _in.size(0) == 1
        _in = _in.to(args.device)
        _out = _out.to(args.device)

        output = model.generate(
            inputs=_in,
            attention_mask=_in.ne(tokenizer.pad_token_id),
            do_sample=False,
            num_beams=2,
            num_return_sequences=1,
            max_new_tokens=len(_out[0]),
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

        pred_ids = output[0][len(_in[0]) :]
        pred = tokenizer.decode(pred_ids, skip_special_tokens=True)
        pred = process_for_eval(pred)
        gt = real_gt[_idx[0]]

        N += 1

        bleu4 += bleu(
            [pred.split()],
            gt.split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=sf,
        )
        if pred.split() == gt.split():
            em += 1
        bar.set_description(f"EM: {em/N}, BLEU: {bleu4/N}")
    logger.info(f"EM: {em/N}, BLEU: {bleu4/N}")

    return em


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tgt_len", default=512, type=int, required=True)
    parser.add_argument("--exemplar_len", default=200, type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True, help="get special tokens")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--data_dir",
        default="../datasets/",
        type=str,
        help="e.g. data_dir/dataset/train.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default="../save",
        type=str,
        help="output dir",
    )
    parser.add_argument("--with_exemplar", type=lambda x: bool(strtobool(x)), required=True)

    parser.add_argument(
        "--model_type",
        default="gpt2",
        type=str,
        help="The model architecture to be fine-tuned.",
    )
    parser.add_argument(
        "--pretrain_dir",
        default="",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        help="config name. Required when training from scratch",
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
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
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
        "--save_steps",
        type=int,
        default=5000,
        help="Save checkpoint every X updates steps.",
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
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
