import json
import os
import pandas as pd
from evaluator.llm_code_eval.llm_evaluator import evaluate_code

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
# model = "gpt-3.5-turbo"
model = "gpt-4"
data_list = []
model_name = model.replace(".","_")
store_path = f"evaluation_res/llm_scores_{model_name}.csv"
print(f"Store path: {store_path}")

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

        # comment the install command
        data = comment_install_command(data)
        # get code-to-code evaluation scores
        # for total test data
        refs = [inst['ground-truth'] for inst in data]
        preds = [inst['prediction'] for inst in data]
        apis = [inst['api'] for inst in data]
        instructions = [inst['instruction'] for inst in data]
        scores = evaluate_code(
            instructions, 
            preds,
            refs,
            apis, 
            new_api_list,
            model=model
        )
        data_obj["usefullness_scores"] = scores[0]
        data_obj["functional_correctness_scores"] = scores[1]
        data_obj["new_apis_usefullness_scores"] = scores[2]
        data_obj["new_apis_functional_correctness_scores"] = scores[3]
        data_obj["langchain_usefullness_scores"] = scores[4]
        data_obj["langchain_functional_correctness_scores"] = scores[5]
        print(data_obj)
        data_list.append(data_obj)
        print("Done............")

        results_df = pd.DataFrame(data_list)

        # Save DataFrame to a CSV file
        results_df.to_csv(store_path, index=False)
        print(f"Saved the results to {store_path}")






