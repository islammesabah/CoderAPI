# @inproceedings{
# zhuo2024icescore,
# title={{ICE}-Score: Instructing Large Language Models to Evaluate Code},
# author={Terry Yue Zhuo},
# booktitle={18th Conference of the European Chapter of the Association for Computational Linguistics},
# year={2024},
# url={https://openreview.net/forum?id=KQ7WB1snxB}
# }
# Licensed under the MIT license.
# code from https://github.com/terryyz/ice-score.git
# llm_code_eval
# with minor modifications

from openai import OpenAI
from collections import Counter
from .utils import TASK_PROMPTS
from tqdm import tqdm
import re

def remove_repeated_patterns(input_string):
    pattern = r'(...................+?)\1+'
    dup_line_pattern = r'^(.*)(\n\1)+$'
    new_text = re.sub(pattern, r'\1', input_string)
    return re.sub(dup_line_pattern, r'\1', new_text, flags=re.MULTILINE)

def get_gpt_answer(raw_content, aspect):
    """
    Extracts the GPT answer from the raw content.
    
    Args:
        raw_content (str): The raw content from GPT response.
        aspect (str): The evaluation aspect.

    Returns:
        int: The extracted answer as an integer.
    """
    try:
        return int(raw_content)
    except ValueError:
        try:
            return process_raw_content(raw_content, aspect)
        except:
            return 0

def process_raw_content(content, aspect):
    """
    Processes the raw content to extract the answer.
    
    Args:
        content (str): The raw content from GPT response.
        aspect (str): The evaluation aspect.

    Returns:
        int: The extracted answer as an integer.
    """
    # Normalize content: lowercase, remove parentheses, and split into lines
    splits = content.lower().replace("(", "").replace(")", "").split("\n")
    
    # Extract lines containing "score", remove dots, and replace "out of" and "/4"
    ls = [
        ll.strip(".").replace("out of ", "/").replace("/4", "")
        for l in splits
        for ll in l.lstrip('0123456789. ').split(". ")
        if any(item in ll for item in ["score"] + aspect.split())
    ]
    
    # Extract all numeric characters in each line and store them in a list
    ans = [ll for l in ls for ll in l.split() if ll.isnumeric()]
    
    # If there are multiple answers, return the most common one
    if len(set(ans)) != 1 and len(ans) > 1:
        return int(Counter(ans).most_common(1)[0][0])
    
    # Handle special cases where there are no answers or multiple non-numeric answers
    if len(set(ans)) != 1:
        if "N/A" in content:
            return 0
            
    # Return the single numeric answer
    return int(ans[0])

def evaluate(problem, output, reference, aspect, model):
    """
    Evaluates the given problem and output using GPT.
    
    Args:
        problem (str): The problem statement.
        output (str): The output of the problem.
        reference (str): The reference solution. Defaults to None.
        aspect (str, optional): The evaluation aspect. Defaults to "usefulness".
        model (str, optional): The GPT model to use. Defaults to "gpt-3.5-turbo".

    Returns:
        int: The evaluation score.
    """
    prompt = TASK_PROMPTS[aspect]
    
    prompt = prompt.replace("{{PROBLEM}}", problem).replace("{{OUTPUT}}", output).replace("{{REFERENCE}}", reference)

    completion = OpenAI().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
    )

    raw_output = completion.choices[0].message.content

    return get_gpt_answer(raw_output, aspect)


def evaluate_code(problems, outputs, references, apis, new_api_list, model="gpt-3.5-turbo"):
    usefullness_scores = []
    functional_correctness_scores = []
    new_apis_usefullness_scores = []
    new_apis_functional_correctness_scores = []
    langchain_usefullness_scores = []
    langchain_functional_correctness_scores = []
    new_apis_count = 0
    langchain_count = 0
    for problem, output, reference, api in tqdm(zip(problems, outputs, references, apis)):
        output = remove_repeated_patterns(output)
        if problem.strip() == "":
            usefullness_score = 0
            functional_correctness_score = 0
        else:
            usefullness_score= evaluate(problem, output, reference, "usefulness", model)
            functional_correctness_score = evaluate(problem, output, reference, "functional correctness", model)
        usefullness_scores.append(usefullness_score)
        functional_correctness_scores.append(functional_correctness_score)
        if api in new_api_list:
            new_apis_count += 1
            new_apis_usefullness_scores.append(usefullness_score)
            new_apis_functional_correctness_scores.append(functional_correctness_score)
        if api == "langchain":
            langchain_count += 1
            langchain_usefullness_scores.append(usefullness_score)
            langchain_functional_correctness_scores.append(functional_correctness_score)
        
    usefullness_score = round(sum(usefullness_scores)/len(problems), 2) if len(problems) > 0 else 0
    functional_correctness_score = round(sum(functional_correctness_scores)/len(problems), 2) if len(problems) > 0 else 0
    new_apis_usefullness_score = round(sum(new_apis_usefullness_scores)/new_apis_count, 2) if new_apis_count > 0 else 0
    new_apis_functional_correctness_score = round(sum(new_apis_functional_correctness_scores)/new_apis_count, 2) if new_apis_count > 0 else 0
    langchain_usefullness_score = round(sum(langchain_usefullness_scores)/langchain_count, 2) if langchain_count > 0 else 0
    langchain_functional_correctness_score = round(sum(langchain_functional_correctness_scores)/langchain_count, 2) if langchain_count > 0 else 0
    
    return (usefullness_score, 
        functional_correctness_score, 
        new_apis_usefullness_score, 
        new_apis_functional_correctness_score, 
        langchain_usefullness_score, 
        langchain_functional_correctness_score)
    