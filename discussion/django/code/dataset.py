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


class exemplarDataset(Dataset):
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
        for idx, x in enumerate(tqdm(data)):
            js = json.loads(x.strip())

            nl = js["nl"]
            exemplars = js["exemplars"]
            code = js["code"]

            nl = tokenizer.tokenize(nl)
            code = tokenizer.tokenize(code)
            exemplars = [tokenizer.tokenize(x) for x in exemplars]

            if n_exemplars != 0 and len(exemplars) == 0:
                continue

            if n_exemplars == 0:
                if "django" in args.dataset:
                    nl = nl[:100]

                nl[0] = "\u0120" + nl[0]
                code[0] = "\u0120" + code[0]

                x = nl + [SEP] + code
                sub_tokens = [SOS] + x[: tgt_len - 2] + [EOS]
                sub_tokens = sub_tokens + [PAD] * (tgt_len - len(sub_tokens))

                assert len(sub_tokens) == tgt_len
                ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                self.inputs.append({"idx": idx, "ids": ids})
                continue

            if len(exemplars) < n_exemplars:
                exemplars += [exemplars[-1]] * (n_exemplars - len(exemplars))
            exemplars = exemplars[:n_exemplars]

            for exemplar in exemplars:
                if "django" in args.dataset:
                    nl = nl[:100]

                exemplar = exemplars[0][:exemplar_len]

                exemplar[0] = "\u0120" + exemplar[0]
                nl[0] = "\u0120" + nl[0]
                code[0] = "\u0120" + code[0]

                if "django" in args.dataset:
                    code = code[:254]
                else:
                    code = code[:382]
                sub_tokens = [SOS] + exemplar + [SEP] + nl + [SEP] + code

                sub_tokens = sub_tokens[: tgt_len - 1] + [EOS]
                sub_tokens = sub_tokens + [PAD] * (tgt_len - len(sub_tokens))

                assert len(sub_tokens) == tgt_len
                ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                if len(ids) == 0:
                    print(ids)
                self.inputs.append({"idx": idx, "ids": ids})

        del data
        gc.collect()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {
            "idx": self.inputs[item]["idx"],
            "ids": torch.tensor(self.inputs[item]["ids"]),
        }
