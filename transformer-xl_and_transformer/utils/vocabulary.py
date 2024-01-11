import os
from collections import Counter, OrderedDict

import torch
from transformers import GPT2Tokenizer

SOS = "<s>"
SEP = "<sep>"
EOS = "</s>"
PAD = "<pad>"
UNK = "<unk>"


class Vocab(object):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def encode_data(self, lines, len_exemplum=256, max_len=512, ordered=False, with_exemplum=True):
        encoded = []
        for line in lines:
            line = line.strip()
            if with_exemplum:
                exemplum = SOS + " " + line.split(SEP)[0][3:].strip()
                context = line.split(SEP)[1][:-4].strip()

                exemplum = self.tokenizer.encode(
                    exemplum,
                    padding=False,
                    truncation=True,
                    max_length=len_exemplum,
                    is_split_into_words=True,
                    add_special_tokens=False,
                )
                assert len(exemplum) <= len_exemplum, "len of exemplum {} != {}".format(
                    len(exemplum), len_exemplum
                )

                context = self.tokenizer.encode(
                    context,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len - len(exemplum) - 1,
                    is_split_into_words=True,
                    add_special_tokens=False,
                )

                try:
                    pad_idx = context.index(self.tokenizer.pad_token_id)
                    context[pad_idx] = self.tokenizer.eos_token_id
                except ValueError:
                    context[-1] = self.tokenizer.eos_token_id

                line = exemplum + [self.tokenizer.convert_tokens_to_ids(SEP)] + context
                assert len(line) == max_len, "len of example {} != {}".format(len(line), max_len)

            else:
                context = SOS + " " + line.split(SEP)[1][:-4].strip()

                context = self.tokenizer.encode(
                    context,
                    padding="max_length",
                    truncation=True,
                    max_length=max_len,
                    is_split_into_words=True,
                    add_special_tokens=False,
                )

                try:
                    pad_idx = context.index(self.tokenizer.pad_token_id)
                    context[pad_idx] = self.tokenizer.eos_token_id
                except ValueError:
                    context[-1] = self.tokenizer.eos_token_id

                line = context
                assert len(line) == max_len, "len of example {} != {}".format(len(line), max_len)

            line = torch.LongTensor(line)
            encoded.append(line)

        if ordered:
            encoded = torch.cat(encoded)
        else:
            encoded = torch.stack(encoded, 0)

        return encoded

    def encode_file(self, path, len_exemplum=256, max_len=512, ordered=False, with_exemplum=True):
        print("encoding file {}".format(path))
        assert os.path.exists(path), "file {} not exists".format(path)

        with open(path, "r") as f:
            lines = f.readlines()

        return self.encode_data(lines, len_exemplum, max_len, ordered, with_exemplum)

    def __len__(self):
        return len(self.tokenizer)


if __name__ == "__main__":
    pass
