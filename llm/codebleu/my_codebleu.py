# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding:utf-8 -*-
import os
import logging
from . import bleu
from . import weighted_ngram_match
from . import syntax_match
from . import dataflow_match


def calc_codebleu(predictions, references, lang, tokenizer=None, params='0.25,0.25,0.25,0.25', kw_dir = "./codebleu", langso_dir = "./codebleu/my-languages.so"):
    """_summary_

    Args:
        predictions (list[str]): list of predictions
        references (list[list[str]): list of lists with references
        lang (str): ['java','js','c_sharp','php','go','python','ruby']
        tokenizer (callable): tokenizer function, Defaults to lambda s: s.split()
        params (str, optional): Defaults to '0.25,0.25,0.25,0.25'.
    """

    alpha, beta, gamma, theta = [float(x) for x in params.split(',')]

    # preprocess inputs
    references = [[x.strip() for x in ref] if type(ref) == list else [ref.strip()] for ref in references]
    hypothesis = [x.strip() for x in predictions]

    if not len(references) == len(hypothesis):
        raise ValueError

    # calculate ngram match (BLEU)
    if tokenizer is None:
        tokenizer = lambda s: s.split()

    tokenized_hyps = [tokenizer(x) for x in hypothesis]
    tokenized_refs = [[tokenizer(x) for x in reference]
                      for reference in references]

    ngram_match_score = bleu.corpus_bleu(tokenized_refs, tokenized_hyps)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open(
        os.path.join(kw_dir, 'keywords', f'{lang}.txt'),
        'r', encoding='utf-8').readlines()]

    def make_weights(reference_tokens, key_word_list):
        return {token: 1 if token in key_word_list else 0.2
                for token in reference_tokens}
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]
                                    for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
        tokenized_refs_with_weights, tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match.corpus_syntax_match(
        references, hypothesis, lang, langso_dir)

    # calculate dataflow match
    dataflow_match_score = dataflow_match.corpus_dataflow_match(
        references, hypothesis, lang, langso_dir)

    # print('ngram match: {0}, weighted ngram match: {1}, syntax_match: {2}, dataflow_match: {3}'.
    #       format(ngram_match_score, weighted_ngram_match_score, syntax_match_score, dataflow_match_score))

    code_bleu_score = alpha*ngram_match_score\
        + beta*weighted_ngram_match_score\
        + gamma*syntax_match_score\
        + theta*(dataflow_match_score or 1)

    # print('CodeBLEU score: ', code_bleu_score)

    return {
        'CodeBLEU': code_bleu_score,
        'ngram_match_score': ngram_match_score,
        'weighted_ngram_match_score': weighted_ngram_match_score,
        'syntax_match_score': syntax_match_score,
        'dataflow_match_score': dataflow_match_score
    }
