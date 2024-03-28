import torch
from transformers import AutoModelForSeq2SeqLM

Best_model = "saved_M6_instruct_codet5p_16b_100epochs_fp16_docu_code_prompt_no_decoder_inp"

model_name = "./saved_models/exp/"+str(Best_model)+"/final_checkpoint_ser"
print("Model Path: ",model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                              torch_dtype=torch.float16,
                                              trust_remote_code=True, 
                                             )

model.push_to_hub("IslamMesabah/CoderAPI", token="<HF_TOKEN_HERE>")