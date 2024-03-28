# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.
# code from https://github.com/microsoft/CodeXGLUE.git
# Code-Code/code-to-code-trans/evaluator/CodeBLEU
# with minor modifications


from .bleu import _bleu
from .CodeBLEU import calc_code_bleu


def evaluate_code_code(refs, preds): 
    assert len(refs) == len(preds)
    
    # calculate exact match accuracy
    exact_match = 0
    for i in range(len(refs)):
        r = refs[i]
        p = preds[i]
        if r.split() == p.split():
            exact_match += 1
    EM = round((exact_match/len(refs))*100, 2)

    # calculate BLEU
    bleu_score = round(_bleu([refs], preds),2)

    # calculate CodeBLEU
    code_bleu_score = round(calc_code_bleu([refs], preds),2)
    
    return  EM, bleu_score, code_bleu_score