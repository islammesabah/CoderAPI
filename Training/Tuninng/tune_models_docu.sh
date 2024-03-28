pip3 install deepspeed

export OMP_NUM_THREADS=4

# Tune models with documentations
sh tune_M5_instruct_codet5p_16b_100epochs_fp16_docu_code_prompt.sh
sh tune_M6_instruct_codet5p_16b_100epochs_fp16_docu_code_prompt_no_decoder_inp.sh
sh tune_M7_instruct_codet5p_16b_100epochs_fp16_general_prompt_docu_encoder_inp.sh
sh tune_M8_instruct_codet5p_16b_100epochs_fp16_code_prompt_docu_encoder_inp.sh
sh tune_M9_instruct_codet5p_16b_100epochs_fp16_docu_code_prompt_docu_encoder_inp.sh
