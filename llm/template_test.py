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
)
from template import _chatgpt_java, api_key
import random
import logging
import argparse

logger = logging.getLogger(__name__)
openai.api_type = "open_ai"


class conf:
    n_shot_list: List[int] = [0, 1]
    search_patterns: dict = {
        # 0: ('header', 'code_tokens'),
        # 1: ('comment', 'code_tokens'),
        2: ("comment", "comment")
    }
    search_tool: str = "lucene"

    n_random: int = 100

    template_index: int = 0

    l_n_tokens: int = 50
    r_n_tokens: int = 300
    output_max_tokens: int = 300

    model_name = "gpt-3.5-turbo"

    seed: int = 123456


def get_response(prompt: List[dict]):
    while True:
        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model=conf.model_name,
                messages=prompt,
                temperature=0,
                max_tokens=conf.output_max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["[END]"],
            )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(e)
            logging.warning(e)
            if str(e).startswith("Rate limit reached") or str(e).startswith(
                "That model is currently overloaded"
            ):
                time.sleep(1)
            continue


def generate_prompt_chatgpt(
    exemplars: List[str], context: str, n_shot: int = 0, len: int = 256
) -> str:
    prompt = []
    user_input = ""
    if n_shot == 0:
        js_system = {
            "role": "system",
            "content": _chatgpt_java[conf.template_index]["0shot_system"].strip(),
        }
        user_input += _chatgpt_java[conf.template_index]["0shot_user_prefix_0"].lstrip()
    else:
        js_system = {
            "role": "system",
            "content": _chatgpt_java[conf.template_index]["nshot_system"].strip(),
        }
        user_input += _chatgpt_java[conf.template_index]["nshot_user_prefix_0"].lstrip()
    prompt.append(js_system)

    for i in range(n_shot):
        idx_brace = exemplars[i].index("{")
        header = exemplars[i][:idx_brace].strip()
        body = exemplars[i][idx_brace:].strip()
        user_input += _chatgpt_java[conf.template_index]["nshot_user_example"].format(
            header=header, body=body
        )

    idx_brace = context.index("{")
    header = context[:idx_brace].strip()
    if n_shot == 0:
        user_input += _chatgpt_java[conf.template_index]["0shot_user_prefix_1"].format(
            header=header
        )
    else:
        user_input += _chatgpt_java[conf.template_index]["nshot_user_prefix_1"].format(
            header=header
        )

    js_user = {"role": "user", "content": user_input.rstrip()}
    prompt.append(js_user)

    return prompt


def get_exemplars_and_context(
    idx: int, pattern: int, search_result: dict, context_dataset: List[dict]
) -> Tuple[List[str], str]:
    exemplars_idx = [item[1] for item in search_result[str(idx)][pattern[0]][pattern[1]][:]]
    context = context_dataset[idx]["code"]
    exemplars = [get_data_from_codebase(idx=i, lang="java")[0] for i in exemplars_idx]

    return exemplars, context


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template_index", "-t", type=int, default=0)
    parser = parser.parse_args()
    conf.template_index = parser.template_index

    conf.log_name: str = f"./logs/template/{conf.template_index}"
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

    search_results_path = f"datasets/search_results/{conf.search_tool}/test_bpe_java.json"
    search_result = json.load(open(search_results_path, "r"))

    context_path = "datasets/save/context_dataset/test_java.jsonl"
    context_dataset = [json.loads(line) for line in open(context_path, "r")]

    filtered_context_idx = []
    for idx, item in enumerate(context_dataset):
        if (
            conf.l_n_tokens
            <= compute_num_tokens(item["code"], model=conf.model_name)
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
                logging.info(prompt[1]["content"])
                logging.info("Prediction:")
                logging.info(pred)
                logging.info("Gt:")
                logging.info(context)

                result_name = f"{n_shot}_{p}"
                gt_header = context.split("{")[0]
                cur_result[result_name] = compute_metric(
                    pred=gt_header + pred, gt=context, lang="java"
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
