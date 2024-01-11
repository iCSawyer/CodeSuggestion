import argparse
import json
import os
import pickle
import re
import sqlite3
from io import BytesIO
from tokenize import (
    COMMENT,
    ENCODING,
    ENDMARKER,
    INDENT,
    NEWLINE,
    NL,
    NUMBER,
    STRING,
    tokenize,
    untokenize,
)
from typing import List, Optional, Tuple


import javalang
from transformers import GPT2TokenizerFast
import argparse


SOS, EOS, PAD, SEP, UNK, EOL = "<s>", "</s>", "<pad>", "<sep>", "<|UNKNOWN|>", "<EOL>"
lit_base_dir = "datasets/save/"


def get_special_tokens(add_sep: bool = True, lang="java"):
    if lang == "python":
        lang = "py"

    lits = json.load(open(lit_base_dir + f"literals_{lang}.json", "r"))
    tokens = ["<STR_LIT>", "<NUM_LIT>", "<CHAR_LIT>"]
    for lit in lits["str"]:
        tokens.append(f"<STR_LIT:{lit}>")
    for lit in lits["num"]:
        tokens.append(f"<NUM_LIT:{lit}>")
    for lit in lits["char"]:
        tokens.append(f"<CHAR_LIT:{lit}>")
    if add_sep:
        tokens.append(SEP)
    return tokens


def ids2tokens(
    tgt_ids: List[int], pred_ids: List[int], tokenizer, rid_0120=True
) -> Tuple[List[str], List[str]]:
    tgt_subtokens = tokenizer.convert_ids_to_tokens(tgt_ids)
    pred_subtokens = tokenizer.convert_ids_to_tokens(pred_ids)
    assert len(tgt_subtokens) == len(pred_subtokens)

    tgt_subs, tgt_tokens, pred_subs, pred_tokens = [], [], [], []
    for tgt_subtoken, pred_subtoken in zip(tgt_subtokens, pred_subtokens):
        if (
            tgt_subtoken in [SOS, SEP, EOS, PAD, UNK, EOL]
            or tgt_subtoken.startswith("<NUM_LIT")
            or tgt_subtoken.startswith("\u0120")
        ):
            if len(tgt_subs) != 0:
                assert len(pred_subs) == len(tgt_subs)
                tgt_tokens.append("".join(tgt_subs))
                tgt_subs = []
                pred_tokens.append("".join(pred_subs))
                pred_subs = []

            if tgt_subtoken.startswith("\u0120"):
                tgt_subs.append(tgt_subtoken)
                pred_subs.append(pred_subtoken)
            else:
                tgt_tokens.append(tgt_subtoken)
                pred_tokens.append(pred_subtoken)
        else:
            tgt_subs.append(tgt_subtoken)
            pred_subs.append(pred_subtoken)

    if len(tgt_subs) != 0:
        tgt_tokens.append("".join(tgt_subs))
        pred_tokens.append("".join(pred_subs))

    assert len(tgt_tokens) == len(pred_tokens)
    if rid_0120:
        _tgt_tokens, _pred_tokens = [], []
        for t, p in zip(tgt_tokens, pred_tokens):
            t = t.replace("\u0120", "")
            p = p.replace("\u0120", "")

            if t == "":
                continue
            else:
                _tgt_tokens.append(t)
                _pred_tokens.append(p)
    else:
        _pred_tokens = pred_tokens
        _tgt_tokens = tgt_tokens

    assert len(_tgt_tokens) == len(_pred_tokens)
    return _tgt_tokens, _pred_tokens


def cal_acc_mrr(preds: List[List[str]], gt: List[str]) -> List[float]:
    assert len(preds) >= 5
    assert all(len(gt) == len(pred) for pred in preds)
    N = len(gt)

    acc1, acc3, acc5 = 0.0, 0.0, 0.0
    mrr1, mrr3, mrr5 = 0.0, 0.0, 0.0

    for tgt_token, *pred_tokens in zip(gt, *preds):
        if tgt_token in [SOS, EOS, PAD, SEP, EOL, UNK]:
            N = N - 1
            continue

        if tgt_token in pred_tokens[:1]:
            acc1 += 1
            mrr1 += 1
            acc3 += 1
            mrr3 += 1
            acc5 += 1
            mrr5 += 1
        elif tgt_token in pred_tokens[:3]:
            cur_rr = 1 / (pred_tokens.index(tgt_token) + 1)
            acc3 += 1
            mrr3 += cur_rr
            acc5 += 5
            mrr5 += cur_rr
        elif tgt_token in pred_tokens[:5]:
            cur_rr = 1 / (pred_tokens.index(tgt_token) + 1)
            acc5 += 1
            mrr5 += cur_rr

    acc1 = acc1 / N
    acc3 = acc3 / N
    acc5 = acc5 / N
    mrr1 = mrr1 / N
    mrr3 = mrr3 / N
    mrr5 = mrr5 / N

    return [acc1, acc3, acc5, mrr1, mrr3, mrr5]


def get_gpt2_tokenizer(
    pretrain_file: str = "microsoft/CodeGPT-small-java", lang="java"
) -> GPT2TokenizerFast:
    assert lang != "python"
    tokenizer = GPT2TokenizerFast.from_pretrained(
        pretrain_file,
        do_lower_case=False,
        sep_token=EOL,
        bos_token=SOS,
        eos_token=EOS,
        pad_token=PAD,
        unk_token=UNK,
        additional_special_tokens=get_special_tokens(lang=lang),
    )
    return tokenizer


def get_data_from_codebase(
    idx: int,
    lang: str,
    conn: Optional[sqlite3.Connection] = None,
    cur: Optional[sqlite3.Cursor] = None,
) -> Tuple[str, str]:
    db_path = f"datasets/codebase_{lang}.db"

    if conn is None or cur is None:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

    sql = "select code, code_tokens, comment, modified_comment, repo, is_valid from codebase where idx = ?"
    cur.execute(sql, (idx,))
    res = cur.fetchall()

    if conn is None or cur is None:
        cur.close()

    return res[0]


def process_string_java(token, special_chars={" ": "U+0020", ",": "U+002C"}):
    lits = json.load(open(lit_base_dir + f"literals_java.json", "r"))
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-z]+"
    qualifier_match = re.search(qualifier_regex, token)

    qualifier = "" if not qualifier_match else qualifier_match[0]

    token_string = re.sub(qualifier_regex, "", token)

    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break
    use_char = False
    if len(str_lit) == 1 and start_quote == "'":
        use_char = True
    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    if not use_char:
        ret = (
            f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
            if str_lit in lits["str"]
            else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
        )
    else:
        ret = (
            f"{qualifier}{start_quote}<CHAR_LIT:{str_lit}>{end_quote}"
            if str_lit in lits["char"]
            else f"{qualifier}{start_quote}<CHAR_LIT>{end_quote}"
        )
    return ret


def preprocess_java(contents: List[str], ignore_exception: bool = False) -> List[str]:
    lits = json.load(open(lit_base_dir + f"literals_java.json", "r"))
    rtn = []
    for content in contents:
        content = content.strip()
        content = re.sub(r"[^\x00-\x7F]", " ", content)
        new_data = []
        try:
            for tok in list(javalang.tokenizer.tokenize(content)):
                if "String" in str(type(tok)) or "Character" in str(type(tok)):
                    token = process_string_java(tok.value)
                elif "Integer" in str(type(tok)) or "FloatingPoint" in str(type(tok)):
                    if tok.value in lits["num"]:
                        token = f"<NUM_LIT:{tok.value}>"
                    else:
                        token = "<NUM_LIT>"
                else:
                    token = tok.value
                new_data.append(token)
            rtn.append(" ".join(new_data))
        except Exception as e:
            if ignore_exception:
                print("warning: add code cannot parsed")
                rtn.append(content)
            else:
                pass
    return rtn


def process_string_py(token, special_chars={" ": "U+0020", ",": "U+002C"}):
    lits = json.load(open(lit_base_dir + f"literals_py.json", "r"))
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-zA-Z]+"
    qualifier_match = re.search(qualifier_regex, token)

    qualifier = "" if not qualifier_match else qualifier_match[0]

    token_string = re.sub(qualifier_regex, "", token)

    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break

    for sc in special_chars:
        str_lit = str_lit.replace(sc, special_chars[sc])
    return (
        f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
        if str_lit in lits["str"]
        else f"{qualifier}{start_quote}<STR_LIT>{end_quote}"
    )


def preprocess_py(contents: List[str], ignore_exception: bool = False):
    lits = json.load(open(lit_base_dir + f"literals_py.json", "r"))
    rtn = []
    for _, source in enumerate(contents):
        try:
            code = source.strip()
            token_gen = tokenize(BytesIO(bytes(code, "utf8")).readline)
            out_tokens = []
            prev_eol = False
            for toknum, tokval, _, _, _ in token_gen:
                tokval = " ".join(tokval.split())
                if toknum == STRING:
                    if tokval.startswith((r"'''", r'"""', r"#")):
                        continue
                    add_token = process_string_py(tokval)
                    out_tokens.append(add_token)
                    prev_eol = False
                elif toknum == NUMBER:
                    if tokval in lits["num"]:
                        out_tokens.append(f"<NUM_LIT:{tokval}>")
                    else:
                        out_tokens.append(f"<NUM_LIT>")
                    prev_eol = False
                elif toknum in [NEWLINE, NL]:
                    if not prev_eol:
                        out_tokens.append("<EOL>")
                        prev_eol = True
                elif toknum in [COMMENT, INDENT, ENCODING, ENDMARKER] or len(tokval) == 0:
                    continue
                else:
                    out_tokens.append(tokval)
                    prev_eol = False
            if out_tokens[0] == "<EOL>":
                out_tokens = out_tokens[1:]
            if out_tokens[-1] == "<EOL>":
                out_tokens = out_tokens[:-1]

            out = " ".join(out_tokens)
            rtn.append(out)

        except Exception as e:
            if ignore_exception:
                print("warning: add to list", flush=True)
                rtn.append(source)
            else:
                pass

    return rtn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", "-l", type=str, default="java")
    parser.add_argument("--idx", "-i", type=int, default=8415632)
    args = parser.parse_args()
    print(get_data_from_codebase(args.idx, args.lang))
