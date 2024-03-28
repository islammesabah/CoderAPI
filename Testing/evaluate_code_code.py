import json
import os
import pandas as pd
from evaluator.evaluator import evaluate_code_code

data_list = pd.read_csv('../Datasets/APIs List.csv')
new_api_list = list(data_list[data_list["Type"] == "New_API"]["Name"])

# read the json data
def read_json_data(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None

# remove exrra colon from the prediction
def remove_extra_colon(data):
    for inst in (data):
        if inst['prediction'][0] == ":":
            inst['prediction'] = inst['prediction'][1:]
    return data

# get test data for the new api list
def get_data_for_new_api(data):
    new_api_data = []
    for inst in data:
        if inst['api'] in new_api_list:
            new_api_data.append(inst)
    return new_api_data

# get test data for the langchain api
def get_data_for_langchain(data):
    langchain_data = []
    for inst in data:
        if "langchain" == inst['api'].lower():
            langchain_data.append(inst)
    return langchain_data

# clean code data 
def comment_install_command(data):
    for inst in data:
        lines = inst['prediction'].split('\n')
        new_lines = []
        for line in lines:
            if "pip install" in line:
                line = "#"+line.strip()
            new_lines.append(line)
        inst['prediction'] = '\n'.join(new_lines)
    return data

# remove python comments
def clean_code_data(data):
    for inst in data:
        lines = inst['ground-truth'].split('\n')
        uncommented_lines = []
        for line in lines:
            line = line.split('#', 1)[0]
            if line.strip() != '':
                line = " ".join(line.strip().split())
                uncommented_lines.append(line)
        inst['ground-truth'] = ' '.join(uncommented_lines)
        lines = inst['prediction'].split('\n')
        uncommented_lines = []
        for line in lines:
            line = line.split('#', 1)[0]
            if line.strip() != '':
                line = " ".join(line.strip().split())
                uncommented_lines.append(line)
        inst['prediction'] = ' '.join(uncommented_lines)
    return data


main_path = "./results/"
tasks = ["Prompt_Engineering",
        "Fine_Tuning"
        ]
data_types = [
    'instructions_general_prompt',
    'instructions_general_prompt_no_decoder_inp',
    'instructions_code_prompt',
    'instructions_code_prompt_no_decoder_inp',
    'instructions_docu_code_prompt',
    'instructions_docu_code_prompt_no_decoder_inp',
    'instructions_general_prompt_docu_encoder_inp',
    'instructions_code_prompt_docu_encoder_inp',
    'instructions_docu_code_prompt_docu_encoder_inp'
    ]

### calculate code-to-code evaluation scores
data_list = []
store_path = "evaluation_res/code_to_code_scores.csv"
if os.path.isfile(store_path):
    data_df = pd.read_csv(store_path) 
    data_list = data_df.to_dict('records')


for task in tasks:
    for data_type in data_types:
        print("\n=====================================")
        print("Task:      ", task, " \nData Type: ", data_type)
        print("=====================================")
        contin = True
        for data in data_list:
            if data["task"] == task and data["data_type"] == data_type:
                print("Already computed")
                contin = False
                continue
        if not contin:
            continue

        data_obj = {}
        data_obj["task"] = task
        data_obj["data_type"] = data_type
        
        path = main_path+task+"/"+data_type+".json"

        # read the data
        data = read_json_data(path)
        if data is None:
            print("No data found for the given path: ", path)
            continue

        data = remove_extra_colon(data)

        # get the tested data for used api and langchain
        new_api_data = get_data_for_new_api(data)
        langchain_data = get_data_for_langchain(data)
        

        # comment the install command
        data = comment_install_command(data)
        new_api_data = comment_install_command(new_api_data)
        langchain_data = comment_install_command(langchain_data)

        # get the compilation score for the data
        # data = clean_code_data(data)
        # new_api_data = clean_code_data(new_api_data)
        # langchain_data = clean_code_data(langchain_data)

        # get code-to-code evaluation scores
        # for total test data
        refs = [inst['ground-truth'] for inst in data]
        preds = [inst['prediction'] for inst in data]
        code_to_code_scores = evaluate_code_code(refs, preds)
        data_obj["total_data_EM"] = code_to_code_scores[0]
        data_obj["total_data_BLEU"] = code_to_code_scores[1]
        data_obj["total_data_codeBLEU"] = code_to_code_scores[2]

        # for new api test data
        refs = [inst['ground-truth'] for inst in new_api_data]
        preds = [inst['prediction'] for inst in new_api_data]
        code_to_code_scores = evaluate_code_code(refs, preds)
        data_obj["new_api_data_EM"] = code_to_code_scores[0]
        data_obj["new_api_data_BLEU"] = code_to_code_scores[1]
        data_obj["new_api_data_codeBLEU"] = code_to_code_scores[2]

        # for langchain test data
        refs = [inst['ground-truth'] for inst in langchain_data]
        preds = [inst['prediction'] for inst in langchain_data]
        code_to_code_scores = evaluate_code_code(refs, preds)
        data_obj["langchain_data_EM"] = code_to_code_scores[0]
        data_obj["langchain_data_BLEU"] = code_to_code_scores[1]
        data_obj["langchain_data_codeBLEU"] = code_to_code_scores[2]

        data_list.append(data_obj)
        print("Done............")
        
results_df = pd.DataFrame(data_list)

# Save DataFrame to a CSV file
results_df.to_csv(store_path, index=False)
print(f"Saved the results to {store_path}")






