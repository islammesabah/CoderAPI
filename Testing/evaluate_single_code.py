import json
from tqdm import tqdm
import os
import pandas as pd
import re
import subprocess

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
    for inst in tqdm(data):
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
        lines = inst['ground-truth'].split('\n')
        new_lines = []
        for line in lines:
            if "pip install" in line:
                line = "# "+line
            new_lines.append(line)
        inst['ground-truth'] = '\n'.join(new_lines)
        lines = inst['prediction'].split('\n')
        new_lines = []
        for line in lines:
            if "pip install" in line:
                line = "# "+line
            new_lines.append(line)
        inst['prediction'] = '\n'.join(new_lines)
    return data

# get the number of instances that uses the api correctly
def get_num_of_codes_uses_api_correctly(data):
    api_count = 0
    for inst in data:
        if inst['api'].lower() in inst["prediction"].lower():
            api_count += 1
    return api_count

# remove the instances with empty prediction
def remove_instant_with_empty_prediction(data):
    empty_prediction = []
    for inst in data:
        if inst['prediction'].strip() == "":
            empty_prediction.append(inst)
    for inst in empty_prediction:
        data.remove(inst)
    return len(empty_prediction), data

# get the compilation score for the given data
def get_compilation_score(data, original_size):
    path = "compilation_testing/code_files/"
    os.makedirs(path, exist_ok=True)
    for i, inst in enumerate(data):
        inst['id'] = i
        os.makedirs(path+"/"+inst["api"], exist_ok=True)
        with open(path+"/"+inst["api"]+"/"+"code_"+str(i)+".py", 'w') as f:
            f.write(inst['prediction'])
    
    # for manual Pylint testing
    # cd compilation_testing
    # sh compile.sh
            
    os.chdir("compilation_testing")
    subprocess.run("bash compile.sh", 
                shell=True,
                stderr=subprocess.DEVNULL
                )
    with open("results.txt", "r") as file:
        results = file.read()
    scores = re.findall(r"Your code has been rated at (\b\d+\.\d+\b)\/10", results)
    compilation_score = 0
    for score in scores:
        compilation_score += float(score) * 10
    os.remove("results.txt")
    os.chdir("..")
    return compilation_score/original_size


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

# ### Compute Compilation score
data_list = []
store_path = "evaluation_res/compilation_score.csv"
if os.path.isfile(store_path):
    data_df = pd.read_csv(store_path) 
    data_list = data_df.to_dict('records')

for task in tasks:
    for data_type in data_types:
        print("\n=====================================")
        print("Task:      ", task, " \nData Type: ", data_type)
        print("=====================================")
        cont = True
        for data in data_list:
            if data["task"] == task and data["data_type"] == data_type:
                print("Already computed")
                cont = False
                continue
        if not cont:
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
        

        # get the count of correctly used api
        count_use_api = get_num_of_codes_uses_api_correctly(data)
        count_use_new_api = get_num_of_codes_uses_api_correctly(new_api_data)
        count_use_langchain = get_num_of_codes_uses_api_correctly(langchain_data)
        original_count = len(data)
        original_count_new_api = len(new_api_data)
        original_count_langchain = len(langchain_data)
        data_obj["correctly_use_of_"+str(original_count)+"_api"] = count_use_api
        data_obj["correctly_use_of_"+str(original_count_new_api)+"_new_api"] = count_use_new_api
        data_obj["correctly_use_of_"+str(original_count_langchain)+"_langchain_api"] = count_use_langchain

        # remove instant with empty prediction and get the count
        empty_prediction_count, data = remove_instant_with_empty_prediction(data)
        empty_count_for_new_api, new_api_data = remove_instant_with_empty_prediction(new_api_data)
        empty_count_for_langchain, langchain_data = remove_instant_with_empty_prediction(langchain_data)
        data_obj["empty_prediction_count"] = empty_prediction_count
        data_obj["empty_prediction_count_new_api"] = empty_count_for_new_api
        data_obj["empty_prediction_count_langchain"] = empty_count_for_langchain

        # get the compilation score for the data
        print("Compilation score for the original data............")
        compilation_score = get_compilation_score(data, original_count)
        print("\nCompilation score for the new api data............")
        compilation_score_new_api = get_compilation_score(new_api_data, original_count_new_api)
        print("\nCompilation score for the langchain data............")
        compilation_score_langchain = get_compilation_score(langchain_data, original_count_langchain)
        data_obj["compilation_score"] = compilation_score
        data_obj["compilation_score_new_api"] = compilation_score_new_api
        data_obj["compilation_score_langchain"] = compilation_score_langchain

        data_list.append(data_obj)
        
results_df = pd.DataFrame(data_list)

# Save DataFrame to a CSV file
results_df.to_csv(store_path, index=False)
print(f"Saved the results to {store_path}")
