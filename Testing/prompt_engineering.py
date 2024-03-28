import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
from tqdm import tqdm
import os

# Hyper-Parameters
device = "cuda" 
max_new_tokens = 1024

os.makedirs('./results/Prompt_Engineering', exist_ok=True)

# helper methods
def get_completion(inst, model, device):
    encoding = {}
    encoding["input_ids"] = torch.tensor(inst["input_ids"]).reshape(1,-1).to(device)
    encoding["attention_mask"] = torch.tensor(inst["attention_mask"]).reshape(1,-1).to(device)
    if inst["decoder_input_ids"] is not None:
        encoding["decoder_input_ids"] = torch.tensor(inst["decoder_input_ids"]).reshape(1,-1).to(device)
    
    with torch.no_grad():
        gen_tokens = model.generate(**encoding,
                                    max_new_tokens = max_new_tokens,
                                    decoder_start_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id)
        if inst["decoder_input_ids"] is not None:
            gen_token = gen_tokens[0][encoding["decoder_input_ids"].shape[-1]:]
        else:
            gen_token = gen_tokens[0]
        return tokenizer.decode(gen_token, skip_special_tokens=True)

def get_data(file_path):
    test_data = load_from_disk(file_path)
    print("File Path: "+file_path)
    print(f'  ==> Loaded {len(test_data)} samples')
    return test_data

model_name = '../Training/saved_models/Salesforce/instructcodet5p-16b'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                              torch_dtype=torch.float16,
                                              low_cpu_mem_usage=True, 
                                              trust_remote_code=True, 
                                              device_map='auto'
                                             )
model.eval()

data_path = "./cache_data/"
data_names = [
    'instructions_general_prompt',
    'instructions_code_prompt',
    'instructions_general_prompt_no_decoder_inp',
    'instructions_code_prompt_no_decoder_inp',
    'instructions_docu_code_prompt',
    'instructions_docu_code_prompt_no_decoder_inp',
    'instructions_general_prompt_docu_encoder_inp',
    'instructions_code_prompt_docu_encoder_inp',
    'instructions_docu_code_prompt_docu_encoder_inp'
]

for data_name in data_names:
    file_path = data_path + data_name
    test_data = get_data(file_path)

    data = []
    for inst in tqdm(test_data):
        out = get_completion(inst, model, device)
        data.append(
            {
                "api":inst["api"],
                "source":inst["source"],
                "decoder_input_ids":True if inst["decoder_input_ids"] is not None else False,
                "instruction":inst["instruction"],
                "ground-truth":inst["output"],
                "prediction":out
            }
        )
        
    with open(f'./results/Prompt_Engineering/{data_name}.json', 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)

