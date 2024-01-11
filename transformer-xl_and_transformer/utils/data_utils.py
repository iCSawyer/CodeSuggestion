import os, sys
import glob
import json

from collections import Counter, OrderedDict
import numpy as np
import torch
import gc
from utils.vocabulary import Vocab
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset
from tqdm import tqdm
from my_utils import *
import pickle

java_omit = [1986, 2761]
py_omit = [
    9,
    11,
    55,
    56,
    192,
    232,
    251,
    909,
    912,
    913,
    915,
    926,
    932,
    935,
    936,
    941,
    947,
    950,
    951,
    952,
    965,
    967,
    994,
    1002,
    1010,
    1031,
    1032,
    1035,
    1036,
    1039,
    1040,
    1051,
    1087,
    1088,
    1097,
    1103,
    1105,
    1108,
    1112,
    1129,
    1133,
    1135,
    1136,
    1141,
    1498,
    1517,
    1526,
    1602,
    1610,
    1615,
    2115,
    2116,
    2117,
    2118,
    2119,
    2122,
    2126,
    2974,
    2975,
    2981,
    2983,
    3123,
    3787,
    3791,
    3792,
    3807,
    3808,
    3809,
    3820,
    3821,
    3823,
    3824,
    3825,
    3834,
    4104,
    4105,
    4115,
    4159,
    4164,
    4409,
]


class exemplarDataset(Dataset):
    def __init__(
        self, tokenizer, args, file_type="valid", exemplar_len=200, tgt_len=512, n_exemplars=1
    ):
        cache_file = os.path.join(
            args.data_dir,
            args.dataset,
            f"{file_type}_{n_exemplars}_{exemplar_len}_{tgt_len}_transformer.pkl",
        )
        if os.path.exists(cache_file):
            pass

        self.inputs = []
        datafile = os.path.join(args.data_dir, args.dataset, f"{file_type}.jsonl")
        if file_type == "train":
            print("Creating features from dataset file at %s", datafile)
        with open(datafile) as f:
            data = f.readlines()
        if args.debug:
            data = data[:10]

        print("Data size: %d, Processing ..." % (len(data)))
        for idx, x in enumerate(tqdm(data)):
            js = json.loads(x.strip())

            if args.lang == "java":
                if js["idx"] in java_omit:
                    continue
            elif args.lang == "py":
                if js["idx"] in py_omit:
                    continue
            else:
                raise NotImplementedError

            context = js["context"]
            exemplars = js["exemplars"]

            context = tokenizer.tokenize(context)
            exemplars = [tokenizer.tokenize(x) for x in exemplars if x != ""]

            if n_exemplars != 0 and len(exemplars) == 0:
                continue

            if n_exemplars == 0:
                sub_tokens = [SOS] + context[: tgt_len - 2] + [EOS]
                sub_tokens = sub_tokens + [PAD] * (tgt_len - len(sub_tokens))

                assert len(sub_tokens) == tgt_len
                ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                self.inputs.append({"idx": idx, "ids": ids})
                continue

            if len(exemplars) < n_exemplars:
                exemplars += [exemplars[-1]] * (n_exemplars - len(exemplars))
            exemplars = exemplars[:n_exemplars]

            for exemplar in exemplars:
                exemplar = exemplar[:exemplar_len]
                sub_tokens = [SOS] + exemplar + [SEP] + context
                sub_tokens = sub_tokens[: tgt_len - 1] + [EOS]
                sub_tokens = sub_tokens + [PAD] * (tgt_len - len(sub_tokens))

                assert len(sub_tokens) == tgt_len
                ids = tokenizer.convert_tokens_to_ids(sub_tokens)
                if len(ids) == 0:
                    print(ids)
                self.inputs.append({"idx": idx, "ids": ids})

        del data
        gc.collect()

        print("%d samples" % (len(self.inputs)))

        pickle.dump(self.inputs, open(cache_file, "wb"))
        print("Save to cache: %s" % cache_file)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        return {"idx": self.inputs[item]["idx"], "ids": torch.tensor(self.inputs[item]["ids"])}


if __name__ == "__main__":
    pass
