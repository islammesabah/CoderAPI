import copy
from datasets import load_dataset
from transformers import AutoTokenizer
import os

PROMPT = {
    'general': (
	"Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    'code': (
	"Below is an instruction that describes a coding task. "
        "Write a full code that appropriately fulfils the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Code:"
    ),
    'docu-code': (
        "This is some documentation of {api} API:\n\n"
        "=====\n"
        "{documentation}\n"
        "=====\n\n"
        "Below is an instruction that describes a coding task. "
        "Write a full code that appropriately completes the request using the above documentation.\n\n"
        "### Instruction:\n{instruction}\n\n### Code:"
    )
}

DOCU_ENCODER_INPUT = {
    "prompt": (
        "This is some documentation of {api} API:\n\n"
        "=====\n"
        "{documentation}\n"
    )
}

def get_fit_in_context(tokenizer, text,max_len):
    input_ids = tokenizer(text,max_length=int(max_len/2), truncation=True)["input_ids"]
    return tokenizer.decode(input_ids, skip_special_tokens=True)

def load_tokenize_test_data(
        cache_data,
        test_data_path,
        tokenizer_path,
        prompt_type='general',
        docu_encoder_input=False,
        max_len=1024,
        with_decoder_input=False
    ):
        datasets = load_dataset('json', data_files=test_data_path)['train']
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        def preprocess_function(examples):
            prompt = PROMPT[prompt_type]
            if prompt_type != 'docu-code':
                source = [prompt.format_map({'instruction': instruct})
                        for instruct in examples["instruction"]]
            else:
                source = [prompt.format_map({'instruction': instruct,
                                             'api': api,
                                             'documentation': get_fit_in_context(tokenizer, documentation, max_len)})
                      for instruct, api, documentation in zip(examples["instruction"],
                                          examples["api"],
                                          examples["documentation"])]
            decoder_inp = tokenizer(source, max_length=max_len, truncation=True)
            decoder_source = source
            # pass documentation to the encoder input
            if docu_encoder_input:
                docu_prompt = DOCU_ENCODER_INPUT["prompt"]
                source = [docu_prompt.format_map({'api': api,
                                             'documentation': documentation})
                      for api, documentation in zip(examples["api"],
                                          examples["documentation"])]

            model_inputs = tokenizer(source, max_length=2048, truncation=True)
            
            if with_decoder_input:
                model_inputs["decoder_input_ids"] = copy.deepcopy(decoder_inp["input_ids"])
                model_inputs["decoder"] = decoder_source
            else:
                model_inputs["decoder_input_ids"] = [None for _ in examples["instruction"]]
                model_inputs["decoder"] = [None for _ in examples["instruction"]]
            model_inputs["source"] = source
            return model_inputs

        test_data = datasets.map(
            preprocess_function,
            batched=True,
            num_proc=64,
            load_from_cache_file=False,
        )

        print(f'  ==> Loaded {len(test_data)} samples')
        test_data.save_to_disk(cache_data)
        print(f'  ==> Saved to {cache_data}')
        return test_data


if __name__ == "__main__":
    # generate testing cache data
    os.makedirs('./cache_data/', exist_ok=True)
    # without documentation
    load_tokenize_test_data(
        cache_data = './cache_data/instructions_general_prompt',
        test_data_path = '../Datasets/pairs_data/cleaned_data/data_docu_test.json' ,
        tokenizer_path = '../Training/saved_models/Salesforce/instructcodet5p-16b',
        prompt_type='general',
        with_decoder_input=True
    )
    load_tokenize_test_data(
        cache_data = './cache_data/instructions_code_prompt',
        test_data_path = '../Datasets/pairs_data/cleaned_data/data_docu_test.json' ,
        tokenizer_path = '../Training/saved_models/Salesforce/instructcodet5p-16b',
        prompt_type='code',
        with_decoder_input=True
    )
    load_tokenize_test_data(
        cache_data = './cache_data/instructions_general_prompt_no_decoder_inp',
        test_data_path = '../Datasets/pairs_data/cleaned_data/data_docu_test.json' ,
        tokenizer_path = '../Training/saved_models/Salesforce/instructcodet5p-16b',
        prompt_type='general'
    )
    load_tokenize_test_data(
        cache_data = './cache_data/instructions_code_prompt_no_decoder_inp',
        test_data_path = '../Datasets/pairs_data/cleaned_data/data_docu_test.json' ,
        tokenizer_path = '../Training/saved_models/Salesforce/instructcodet5p-16b',
        prompt_type='code'
    )

    # with documentation
    load_tokenize_test_data(
        cache_data = './cache_data/instructions_docu_code_prompt',
        test_data_path = '../Datasets/pairs_data/cleaned_data/data_docu_test.json' ,
        tokenizer_path = '../Training/saved_models/Salesforce/instructcodet5p-16b',
        prompt_type='docu-code',
        with_decoder_input=True
    )
    load_tokenize_test_data(
        cache_data = './cache_data/instructions_docu_code_prompt_no_decoder_inp',
        test_data_path = '../Datasets/pairs_data/cleaned_data/data_docu_test.json' ,
        tokenizer_path = '../Training/saved_models/Salesforce/instructcodet5p-16b',
        prompt_type='docu-code'
    )
    load_tokenize_test_data(
        cache_data = './cache_data/instructions_general_prompt_docu_encoder_inp',
        test_data_path = '../Datasets/pairs_data/cleaned_data/data_docu_test.json' ,
        tokenizer_path = '../Training/saved_models/Salesforce/instructcodet5p-16b',
        prompt_type='general',
        with_decoder_input=True,
        docu_encoder_input=True
    )
    load_tokenize_test_data(
        cache_data = './cache_data/instructions_code_prompt_docu_encoder_inp',
        test_data_path = '../Datasets/pairs_data/cleaned_data/data_docu_test.json' ,
        tokenizer_path = '../Training/saved_models/Salesforce/instructcodet5p-16b',
        prompt_type='code',
        with_decoder_input=True,
        docu_encoder_input=True
    )
    load_tokenize_test_data(
        cache_data = './cache_data/instructions_docu_code_prompt_docu_encoder_inp',
        test_data_path = '../Datasets/pairs_data/cleaned_data/data_docu_test.json' ,
        tokenizer_path = '../Training/saved_models/Salesforce/instructcodet5p-16b',
        prompt_type='docu-code',
        with_decoder_input=True,
        docu_encoder_input=True
    )
