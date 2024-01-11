import argparse
import json
import os
from tokenize import NAME, OP, tokenize

from my_utils import *


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


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--lang", type=str, default="java")
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--remove", type=str, default="yes")
    args = args.parse_args()

    lang = args.lang
    pred_token_path = "_a.json"
    tgt_token_path = f"../../datasets/{args.dataset}/test.jsonl"

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

    pred_tokens = [item[0] for item in json.load(open(pred_token_path))]

    with open(tgt_token_path) as f:
        lines = f.readlines()
        lines = [json.loads(line.strip()) for line in lines]
    if lang == "py":
        tgt_tokens = [item["context"].split(" ") for item in lines if item["idx"] not in py_omit]
    else:
        tgt_tokens = [item["context"].split(" ") for item in lines if item["idx"] not in java_omit]

    assert len(pred_tokens) == len(tgt_tokens), f"{len(pred_tokens)}, {len(tgt_tokens)}"
    print(pred_tokens[0])
    if args.remove == "yes":
        pred_tokens = [item[1:] for item in pred_tokens]
    pred_tokens = [item[: len(tgt_tokens[idx])] for idx, item in enumerate(pred_tokens)]
    print(pred_tokens[0])
    print(tgt_tokens[0])

    _result = []
    result = {}
    total_token, correct_token = 0, 0
    for list_x, list_y in zip(pred_tokens, tgt_tokens):
        if len(list_x) < len(list_y):
            list_x += [""] * (len(list_y) - len(list_x))

        _total, _right = 0, 0
        for x, y in zip(list_x, list_y):
            x = x.replace("\u0120", "")
            y = y.replace("\u0120", "")
            if y not in [SOS, EOS, SEP, PAD, EOL, UNK, ""]:
                total_token += 1
                _total += 1
                try:
                    typ = judge(lang, y)
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
                result[typ] = result.get(typ, (0, 0))
                if x == y:
                    correct_token += 1
                    _right += 1
                    result[typ] = (result[typ][0] + 1, result[typ][1] + 1)
                else:
                    result[typ] = (result[typ][0], result[typ][1] + 1)
        _result.append(_right / _total)

    json.dump(_result, open("result.json", "w"))
    print(len(_result))

    for key in result:
        print(
            f"Type:{key:12s}, Part:{result[key][1]/total_token:6.5f}, Acc:{result[key][0]/result[key][1]:6.5f}"
        )
    print(
        f"Type:Overall, Part:{total_token/total_token:6.5f}, Acc:{correct_token/total_token:6.5f}"
    )
