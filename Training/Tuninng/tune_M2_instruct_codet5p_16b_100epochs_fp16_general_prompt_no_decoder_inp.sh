torchrun --nproc-per-node 4 ../instruct_tune_codet5p.py \
    --instruct-data-path '../../Datasets/pairs_data/cleaned_data/data_docu_train.json' \
    --load '../saved_models/Salesforce/instructcodet5p-16b' \
    --deepspeed '../deepspeed_config.json' \
    --cache-data '../cache_data/instructions_general_prompt_no_decoder_inp' \
    --save-dir "../saved_models/exp/saved_M2_instruct_codet5p_16b_100epochs_fp16_general_prompt_no_decoder_inp" \
    --epochs 100 \
    --fp16 \
    --lr 2e-5 \
    --prompt-type 'general'
