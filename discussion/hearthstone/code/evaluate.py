from __future__ import absolute_import, division, print_function

import argparse
import json
import logging
import os
import re
import warnings
from distutils.util import strtobool

import debugpy
import numpy as np
import torch
from dataset import trainDataset, testDataset
from my_utils import *
from nltk import bleu
from nltk.translate.bleu_score import SmoothingFunction
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

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

    logger.warning(f"Loading {file_type} examples with {n_exemplars} exemplars")

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

    return dataset


def eval_acc(args, model, tokenizer, file_type="test"):
    eval_dataset = load_and_cache_examples(args, tokenizer, file_type="test")
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
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
        idx, ids = batch["idx"], batch["ids"]
        assert ids.size(0) == 1
        ids = ids.to(args.device)

        start_pos = ids[0].tolist()[::-1].index(tokenizer.convert_tokens_to_ids("<sep>"))
        start_pos = len(ids[0]) - start_pos - 1
        input_ids = ids[:, : start_pos + 1]

        gt_ids = ids[:, start_pos + 1 :].cpu().tolist()[0]
        gt_len = len(gt_ids)

        output = model.generate(
            inputs=input_ids,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            do_sample=False,
            num_beams=10,
            num_return_sequences=1,
            max_new_tokens=gt_len,
            pad_token_id=tokenizer.eos_token_id,
            early_stopping=True,
        )

        pred_ids = output[0][start_pos + 1 :].cpu().tolist()

        pred = tokenizer.decode(pred_ids)
        pred_eos_pos = pred.find("</s>")
        pred = pred[:pred_eos_pos]
        pred = process_for_eval(pred)

        N += 1
        print(pred)
        print(real_gt[step])

        bleu4 += bleu(
            [pred.lower().split()],
            real_gt[step].lower().split(),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=sf,
        )
        if pred.split() == real_gt[step].split():
            em += 1
        bar.set_description(f"EM: {em / N}, BLEU: {bleu4 / N}")

    print(f"EM: {em / N}, BLEU: {bleu4 / N}")


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
            final_preds = eval_acc(args, model, tokenizer)
        datafile = os.path.join(args.data_dir, args.dataset, "test.jsonl")
        with open(datafile, "r") as f:
            lines = f.readlines()
            gt = [process_for_eval(json.loads(line.strip())["code"]) for line in lines]

        N = len(final_preds)

        em, bleu4 = 0, 0
        for i in range(N):
            sf = SmoothingFunction().method2
            bleu4 += bleu(
                [final_preds[i].lower().split()],
                gt[i].lower().split(),
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=sf,
            )
            if final_preds[i].split() == gt[i].split():
                em += 1
        em /= N
        bleu4 /= N

        logger.info(f"EM: {em}, BLEU: {bleu4}")


if __name__ == "__main__":
    main()
