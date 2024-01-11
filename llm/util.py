import io
import sqlite3
import tokenize
from typing import List, Optional, Tuple, Union

import javalang
import tiktoken
from Levenshtein import distance, ratio
from nltk import bleu

from codebleu.my_codebleu import calc_codebleu
from nltk.translate.bleu_score import SmoothingFunction
import re


def remove_comments_and_docstrings(source, lang):
    if lang in ["py"]:
        """
        Returns 'source' minus comments and docstrings.
        """
        io_obj = io.StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += " " * (start_col - last_col)
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split("\n"):
            if x.strip() != "":
                temp.append(x)
        return "\n".join(temp)
    elif lang in ["ruby"]:
        return source
    else:

        def replacer(match):
            s = match.group(0)
            if s.startswith("/"):
                return " "  # note: a space and not an empty string
            else:
                return s

        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"', re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split("\n"):
            if x.strip() != "":
                temp.append(x)
        return "\n".join(temp)


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


def get_header_and_body(source):
    lines = source.split("\n")

    colon_idx = -1
    for idx, line in enumerate(lines):
        line = line.strip()

        if len(line) == 0:
            continue

        if line[-1] == ":":
            colon_idx = idx
            break

    if colon_idx == -1:
        raise Exception

    header = "\n".join(lines[: colon_idx + 1])
    body = "\n".join(lines[colon_idx + 1 :])

    return header, body


def split_to_tokens(input: str, lang="java") -> Optional[List[str]]:
    if lang == "java":
        try:
            tokens = list(javalang.tokenizer.tokenize(input, ignore_errors=True))
            return [token.value for token in tokens], True
        except:
            return input.split(), False

    elif lang == "py":
        try:
            code_tokens = []
            tup_tokens = list(tokenize.tokenize(io.BytesIO(input.encode("utf-8")).readline))
            for tup in tup_tokens:
                if (
                    tup[0]
                    in [
                        tokenize.COMMENT,
                        tokenize.ENCODING,
                        tokenize.NL,
                        tokenize.NEWLINE,
                        tokenize.INDENT,
                        tokenize.DEDENT,
                        tokenize.ENDMARKER,
                    ]
                    or len(tup[1]) == 0
                    or tup[1].startswith(("'''", '"""', "#"))
                ):
                    continue
                code_tokens.append(tup[1])
            return code_tokens, True
        except:
            return input.split(), False


def compute_codebleu(pred, gt, lang: str = "java") -> dict:
    if lang == "py":
        lang = "python"
    res = calc_codebleu(predictions=[pred], references=[[gt]], lang=lang)
    return res["CodeBLEU"]


def compute_metric(pred: str, gt: str, lang="java") -> dict:
    try:
        pred = remove_comments_and_docstrings(pred, lang)
        gt = remove_comments_and_docstrings(gt, lang)
    except:
        pass

    pred_tokens, pred_flag = split_to_tokens(pred, lang=lang)
    gt_tokens, gt_flag = split_to_tokens(gt, lang=lang)

    if not pred_flag or not gt_flag:
        print("WARNING")
        pred_tokens = pred.split()
        gt_tokens = gt.split()

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

    pred_tokens = f(" ".join(pred_tokens))
    gt_tokens = f(" ".join(gt_tokens))
    pred = " ".join(pred_tokens)
    gt = " ".join(gt_tokens)

    sf = SmoothingFunction().method2
    result = {
        "ES": ratio(gt, pred),
        "EM": 1 if gt == pred else 0,
        "bleu1": bleu([gt_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=sf),
        "bleu2": bleu([gt_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=sf),
        "bleu3": bleu(
            [gt_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=sf
        ),
        "bleu4": bleu(
            [gt_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf
        ),
        "codebleu": compute_codebleu(pred, gt, lang=lang),
        "N": 1,
    }

    return result


def show_result(result: dict) -> None:
    res = {}
    for k in result:
        res[k] = format(result[k], ".3f")
    return res


def add_two_results(r1: dict, r2: dict) -> dict:
    res = {}
    for k in r1:
        res[k] = r1[k] + r2[k]
    return res


def divide_result(result: dict) -> dict:
    res = {}

    n = result["N"]
    for k in result:
        if k != "N":
            res[k] = result[k] / n
        else:
            res[k] = result[k]

    return res


def compute_num_tokens(input: str, model: str) -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(input))
