pip3 install deepspeed

export OMP_NUM_THREADS=4

# Tune models without documentations
sh tune_M1_instruct_codet5p_16b_100epochs_fp16_general_prompt.sh                        # 11:26:27
sh tune_M2_instruct_codet5p_16b_100epochs_fp16_general_prompt_no_decoder_inp.sh  # 11:07:21
sh tune_M3_instruct_codet5p_16b_100epochs_fp16_code_prompt.sh                           # 11:08:29
sh tune_M4_instruct_codet5p_16b_100epochs_fp16_code_prompt_no_decoder_inp.sh
