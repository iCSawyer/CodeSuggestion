from __future__ import absolute_import, division, print_function
import gc
import json
import os
import torch
from my_utils import *
from torch.utils.data import (
    Dataset,
)
from tqdm import tqdm


class trainDataset(Dataset):
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
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        self.inputs = []
        datafile = os.path.join(args.data_dir, args.dataset, f"{file_type}.jsonl")
        if file_type == "train":
            logger.warning("Creating features from dataset file at %s", datafile)
        with open(datafile) as f:
            data = f.readlines()

        logger.info("Data size: %d, Processing ..." % (len(data)))
        logger.warning("FORMAT: EXEMPLAR + NL + CODE")
        for idx_x, x in enumerate(tqdm(data)):
            js = json.loads(x.strip())
            nl = js["nl"]
            exemplars = js["exemplars"]
            code = js["code"]

            nl = tokenizer.tokenize(nl)
            code = tokenizer.tokenize(code)
            exemplar = tokenizer.tokenize(exemplars[0])

            if n_exemplars == 0:
                assert args.with_exemplar == False

                nl[0] = "\u0120" + nl[0]
                code[0] = "\u0120" + code[0]
                x = nl + [SEP] + code
                sub_tokens = [SOS] + x[: tgt_len - 2] + [EOS]
                sub_tokens = sub_tokens + [PAD] * (tgt_len - len(sub_tokens))
                assert len(sub_tokens) == tgt_len

                ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                self.inputs.append({"idx": idx_x, "ids": ids})
            else:
                assert args.with_exemplar == True

                nl[0] = "\u0120" + nl[0]
                code[0] = "\u0120" + code[0]
                exemplar[0] = "\u0120" + exemplar[0]

                exemplar = exemplar + [SEP]
                exemplar = exemplar[:exemplar_len]

                x = nl + [SEP] + code
                x = x[: 384 - 2]
                sub_tokens = [SOS] + exemplar + x + [EOS]
                sub_tokens = sub_tokens + [PAD] * (tgt_len - len(sub_tokens))
                assert len(sub_tokens) == tgt_len

                ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                self.inputs.append({"idx": idx_x, "ids": ids})

        del data
        gc.collect()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {
            "idx": self.inputs[item]["idx"],
            "ids": torch.tensor(self.inputs[item]["ids"]),
        }


class testDataset(Dataset):
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
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        self.inputs = []
        datafile = os.path.join(args.data_dir, args.dataset, f"{file_type}.jsonl")
        if file_type == "train":
            logger.warning("Creating features from dataset file at %s", datafile)
        with open(datafile) as f:
            data = f.readlines()

        logger.info("Data size: %d, Processing ..." % (len(data)))
        logger.warning("FORMAT: EXEMPLAR + NL + CODE")
        for idx_x, x in enumerate(tqdm(data)):
            js = json.loads(x.strip())
            nl = js["nl"]
            exemplars = js["exemplars"]
            code = js["code"]

            nl = tokenizer.tokenize(nl)
            code = tokenizer.tokenize(code)
            exemplar = tokenizer.tokenize(exemplars[0])

            if n_exemplars == 0:
                assert args.with_exemplar == False

                nl[0] = "\u0120" + nl[0]
                code[0] = "\u0120" + code[0]

                _in = [SOS] + nl + [SEP]
                _out = code + [EOS]

                _in = tokenizer.convert_tokens_to_ids(_in)
                _out = tokenizer.convert_tokens_to_ids(_out)

                self.inputs.append({"idx": idx_x, "in": _in, "out": _out})
            else:
                assert args.with_exemplar == True

                nl[0] = "\u0120" + nl[0]
                code[0] = "\u0120" + code[0]
                exemplar[0] = "\u0120" + exemplar[0]

                _in = [SOS] + exemplar + [SEP] + nl + [SEP]
                _out = code + [EOS]
                _in = tokenizer.convert_tokens_to_ids(_in)
                _out = tokenizer.convert_tokens_to_ids(_out)

                self.inputs.append({"idx": idx_x, "in": _in, "out": _out})

        del data
        gc.collect()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {
            "idx": self.inputs[item]["idx"],
            "in": torch.tensor(self.inputs[item]["in"]),
            "out": torch.tensor(self.inputs[item]["out"]),
        }
