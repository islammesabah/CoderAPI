# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
# code from https://github.com/microsoft/CodeXGLUE.git
# Code-Code/code-to-code-trans/evaluator/CodeBLEU
# with minor modifications

from .bleu import corpus_bleu as bleu
from .weighted_ngram_match import corpus_bleu as weighted_ngram_match
from .syntax_match import corpus_syntax_match as syntax_match
from .dataflow_match import corpus_dataflow_match as dataflow_match

def calc_code_bleu(pre_references, hypothesis, params=[0.25,0.25,0.25,0.25]):
    alpha,beta,gamma,theta = params

    # preprocess inputs
    for i in range(len(pre_references)):
        assert len(hypothesis) == len(pre_references[i])

    references = []
    for i in range(len(hypothesis)):
        ref_for_instance = []
        for j in range(len(pre_references)):
            ref_for_instance.append(pre_references[j][i])
        references.append(ref_for_instance)
    assert len(references) == len(pre_references)*len(hypothesis)


    # calculate ngram match (BLEU)
    tokenized_hyps = [x.split() for x in hypothesis]
    tokenized_refs = [[x.split() for x in reference] for reference in references]

    ngram_match_score = bleu(tokenized_refs,tokenized_hyps)

    # calculate weighted ngram match
    keywords = [x.strip() for x in open('evaluator/CodeBLEU/keywords/python.txt', 'r', encoding='utf-8').readlines()]
    def make_weights(reference_tokens, key_word_list):
        return {token:1 if token in key_word_list else 0.2 \
                for token in reference_tokens}
    tokenized_refs_with_weights = [[[reference_tokens, make_weights(reference_tokens, keywords)]\
                for reference_tokens in reference] for reference in tokenized_refs]

    weighted_ngram_match_score = weighted_ngram_match(tokenized_refs_with_weights,tokenized_hyps)

    # calculate syntax match
    syntax_match_score = syntax_match(references, hypothesis)

    # calculate dataflow match
    dataflow_match_score = dataflow_match(references, hypothesis)
    
    code_bleu_score = alpha*ngram_match_score\
                    + beta*weighted_ngram_match_score\
                    + gamma*syntax_match_score\
                    + theta*dataflow_match_score

    return round((code_bleu_score * 100),2)




