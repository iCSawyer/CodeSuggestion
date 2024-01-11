from typing import List, Tuple
import json
import os
import openai
import time
import os
from util import (
    get_data_from_codebase,
    compute_metric,
    add_two_results,
    divide_result,
    show_result,
    compute_num_tokens,
    get_header_and_body,
)
from template import _chatgpt_py, api_key
import random
import logging
import argparse
from fuzzywuzzy import fuzz


logger = logging.getLogger(__name__)
openai.api_type = "open_ai"
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


class conf:
    n_shot_list: List[int] = [0, 1]
    search_patterns: dict = {
        0: ("header", "code_tokens"),
        1: ("comment", "code_tokens"),
        2: ("comment", "comment"),
    }
    search_tool: str = "dense_llm"

    n_random: int = 500

    template_index: int = -1

    l_n_tokens: int = 50
    r_n_tokens: int = 300
    output_max_tokens: int = 300

    model_name = "text-davinci-003"

    seed: int = 123456


def get_response(prompt: List[dict]):
    while True:
        try:
            openai.api_key = api_key
            response = openai.Completion.create(
                engine=conf.model_name,
                prompt=prompt,
                temperature=0,
                max_tokens=conf.output_max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["[END]"],
            )
            return response["choices"][0].text.strip()
        except Exception as e:
            print(e)
            logging.warning(e)
            if str(e).startswith("Rate limit reached") or str(e).startswith(
                "That model is currently overloaded"
            ):
                time.sleep(3)
            continue


def generate_prompt_chatgpt(
    exemplars: List[str], context: str, n_shot: int = 0, length: int = 512
) -> str:
    prompt = []
    user_input = ""
    if n_shot == 0:
        js_system = {
            "role": "system",
            "content": _chatgpt_py[conf.template_index]["0shot_system"].strip(),
        }
        user_input += _chatgpt_py[conf.template_index]["0shot_user_prefix_0"].lstrip()
    else:
        js_system = {
            "role": "system",
            "content": _chatgpt_py[conf.template_index]["nshot_system"].strip(),
        }
        user_input += _chatgpt_py[conf.template_index]["nshot_user_prefix_0"].lstrip()
    prompt.append(js_system)

    for i in range(n_shot):
        header, body = get_header_and_body(exemplars[min(i, len(exemplars) - 1)])
        header, body = header.strip(), body.rstrip()[:length]
        user_input += _chatgpt_py[conf.template_index]["nshot_user_example"].format(
            header=header, body=body
        )

    header, _ = get_header_and_body(context)
    header = header.strip()
    if n_shot == 0:
        user_input += _chatgpt_py[conf.template_index]["0shot_user_prefix_1"].format(header=header)
    else:
        user_input += _chatgpt_py[conf.template_index]["nshot_user_prefix_1"].format(header=header)

    js_user = {"role": "user", "content": user_input.rstrip()}
    return user_input.rstrip()


def get_exemplars_and_context(
    idx: int, pattern: tuple, search_result: dict, context_dataset: List[dict]
) -> Tuple[List[str], str]:
    def is_overlap(repo1, repo2):
        repo1 = repo1.strip().lower()
        repo2 = repo2.strip().lower()

        import re

        rule = r"""/|-|_| |\.|0|1|2|3|4|5|6|7|8|9|python|py|java"""
        repo1_list = " ".join(re.split(rule, repo1)).split()
        repo2_list = " ".join(re.split(rule, repo2)).split()
        repo1 = "".join(repo1_list).strip()
        repo2 = "".join(repo2_list).strip()

        if not set(repo1_list) & set(repo2_list) and not fuzz.partial_ratio(repo1, repo2) == 100:
            return False
        return True

    exemplars_idx = [item[1] for item in search_result[str(idx)][pattern[0]][pattern[1]][:]]
    context = context_dataset[idx]["code"]
    exemplars = []
    c_repo = context_dataset[idx]["repo"]
    for i in exemplars_idx:
        exemplar, _, _, _, e_repo, _ = get_data_from_codebase(idx=i, lang="py")
        if not is_overlap(e_repo, c_repo):
            exemplars.append(exemplar)
    return exemplars, context


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_index", "-t", type=int, required=True)
    parser = parser.parse_args()
    conf.template_index = parser.template_index

    conf.log_name: str = f"./logs/davinci/py_{conf.template_index}_{conf.search_tool}"
    conf.log_name += time.strftime("_time%m%d-%H%M%S.log", time.localtime())
    logging.basicConfig(
        filename=conf.log_name,
        format="%(asctime)s %(message)s",
        datefmt="%m/%d/ %H:%M:%S",
        level=logging.INFO,
    )

    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    logging.info([f"{k}: {v}" for k, v in vars(conf).items() if not k.startswith("__")])

    search_results_path = f"datasets/search_results/{conf.search_tool}/test_bpe_py.json"
    search_result = json.load(open(search_results_path, "r"))

    context_path = "datasets/save/context_dataset/test_py.jsonl"
    context_dataset = [json.loads(line) for line in open(context_path, "r")]

    filtered_context_idx = []
    for idx, item in enumerate(context_dataset):
        if idx in py_omit:
            continue
        if (
            conf.l_n_tokens
            <= compute_num_tokens(item["code"], model="gpt-3.5-turbo")
            <= conf.r_n_tokens
        ):
            filtered_context_idx.append(idx)

    random.seed(conf.seed)
    samples_idx = random.sample(range(len(filtered_context_idx)), conf.n_random)
    samples = [filtered_context_idx[idx] for idx in samples_idx]

    results = {}
    for idx in samples:
        logger.info("=====================================")
        logger.info(f"idx: {idx}")
        assert (
            idx == context_dataset[idx]["idx"]
        ), f"idx: {idx}, context_dataset[idx]['idx']: {context_dataset[idx]['idx']}"

        cur_result = {}

        for n_shot in conf.n_shot_list:
            for i, p in enumerate(conf.search_patterns):
                if n_shot == 0 and i != 0:
                    break

                exemplars, context = get_exemplars_and_context(
                    idx, conf.search_patterns[p], search_result, context_dataset
                )
                prompt = generate_prompt_chatgpt(exemplars, context, n_shot)
                pred = get_response(prompt)

                time.sleep(1)
                logging.info(f"========== (n_shot)_(search_pattern): {n_shot}_{p} ==========")
                logging.info("Prompt:")
                logging.info(prompt)
                logging.info("Prediction:")
                logging.info(pred)
                logging.info("Gt:")
                logging.info(context)

                result_name = f"{n_shot}_{p}"
                gt_header, _ = get_header_and_body(context)
                cur_result[result_name] = compute_metric(
                    pred=gt_header + pred, gt=context, lang="py"
                )

        for result_name in cur_result:
            if result_name not in results:
                results[result_name] = {
                    "ES": 0.0,
                    "EM": 0.0,
                    "bleu1": 0.0,
                    "bleu2": 0.0,
                    "bleu3": 0.0,
                    "bleu4": 0.0,
                    "codebleu": 0.0,
                    "N": 0,
                }
            results[result_name] = add_two_results(results[result_name], cur_result[result_name])

            logger.info(
                f"(n_shot)_(q_type):{result_name}:\n{show_result(divide_result(results[result_name]))}"
            )

        logger.info("\n")
