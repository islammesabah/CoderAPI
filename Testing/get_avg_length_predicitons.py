import json
from tqdm import tqdm
import os
import pandas as pd
import re

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

# get the avg length of the predictions and the avg time per word
def get_avg_time_per_each_word(data, time_h, time_m):
    total_length = 0
    for inst in data:
        total_length += len(inst['prediction'].split())
    time_sec = time_h*3600 + time_m*60
    return (time_sec / total_length, total_length/len(data))
    
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
store_path = "evaluation_res/time_predictions.csv"
if os.path.isfile(store_path):
    data_df = pd.read_csv(store_path) 
    data_list = data_df.to_dict('records')

new_data_list = []
for task in tasks:
    for data_type in data_types:
        print("\n=====================================")
        print("Task:      ", task, " \nData Type: ", data_type)
        print("=====================================")
        
        for data in data_list:
            if data["task"] == task and data["data_type"] == data_type:
                time_h = data["hours"]
                time_m = data["minutes"]
                continue
        
        data_obj = {
            "task": task,
            "data_type": data_type,
            "hours": time_h,
            "minutes": time_m
        }
        
        path = main_path+task+"/"+data_type+".json"
        # read the data
        data = read_json_data(path)
        if data is None:
            print("No data found for the given path: ", path)
            continue

        data = remove_extra_colon(data)

        # get the avg length of the predictions
        avg_time, avg_length = get_avg_time_per_each_word(data,time_h, time_m)
        data_obj["avg_time"] = avg_time
        data_obj["avg_length"] = avg_length

        new_data_list.append(data_obj)

results_df = pd.DataFrame(new_data_list)

store_path = "evaluation_res/avg_time_predictions.csv"
# Save DataFrame to a CSV file
results_df.to_csv(store_path, index=False)
print(f"Saved the results to {store_path}")
