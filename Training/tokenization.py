import copy
import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

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
    input_ids = tokenizer(text,max_length=int(max_len/4), truncation=True)["input_ids"]
    return tokenizer.decode(input_ids, skip_special_tokens=True)

def load_tokenize_data(args):
    # Load and tokenize data
    if os.path.exists(args.cache_data):
        train_data = load_from_disk(args.cache_data)
        print(f'  ==> Loaded {len(train_data)} samples')
        return train_data
    else:
        datasets = load_dataset('json', data_files=args.instruct_data_path)['train']
        tokenizer = AutoTokenizer.from_pretrained(args.load)

        def preprocess_function(examples):
            prompt = PROMPT[args.prompt_type]
            if args.prompt_type != 'docu-code':
                source = [prompt.format_map({'instruction': instruct})
                      for instruct in examples["instruction"]]
            else:
                source = [prompt.format_map({'instruction': instruct,
                                             'api': api,
                                             'documentation': get_fit_in_context(tokenizer, documentation, args.max_len)})
                      for instruct, api, documentation in zip(examples["instruction"],
                                          examples["api"],
                                          examples["documentation"])]
            
            # basic input tokenization
            model_inputs = tokenizer(source, max_length=2048,truncation=True)
            eos_token_id = tokenizer.eos_token_id

            # labels tokenization
            if args.with_decoder_input:
                target = [src + output + tokenizer.eos_token for src, output in zip(source, examples["output"])]
                labels = tokenizer(target, max_length=args.max_len, padding="max_length", truncation=True)
                labels_ids = copy.deepcopy(labels["input_ids"])
                # changing labels: convert all tokens in the duplicate prefix prompt to -100
                for x, y in zip(model_inputs["input_ids"], labels["input_ids"]):
                    label_prefix_len = x.index(eos_token_id) if eos_token_id in x else len(x)
                    y[:label_prefix_len] = [-100] * label_prefix_len
            else:
                target = [output + tokenizer.eos_token for output in examples["output"]]
                labels = tokenizer(target, max_length=args.max_len, padding="max_length", truncation=True)
                labels_ids = copy.deepcopy(labels["input_ids"])

            # changing labels: convert all tokens in the the padding part to -100
            for y in labels["input_ids"]:
                if eos_token_id in y:
                    pad_len = len(y) - y.index(eos_token_id) - 1
                    if pad_len > 0:
                        y[y.index(eos_token_id) + 1:] = [-100] * pad_len

            # pass documentation to the encoder input
            if args.docu_encoder_input:
                docu_prompt = DOCU_ENCODER_INPUT["prompt"]
                source = [docu_prompt.format_map({'api': api,
                                             'documentation': documentation})
                      for api, documentation in zip(examples["api"],
                                          examples["documentation"])]

                model_inputs = tokenizer(source, max_length=2048,truncation=True)

            # decoder input
            model_inputs["decoder_input_ids"] = copy.deepcopy(labels_ids)

            # shift labels to the right as the decoder input and add decoder start token id for training
            decoder_start_id = tokenizer.eos_token_id
            for z in model_inputs["decoder_input_ids"]:
                z[1:] = z[:-1]
                z[0] = decoder_start_id

            model_inputs["labels"] = copy.deepcopy(labels["input_ids"])
            model_inputs["decoder_attention_mask"] = labels["attention_mask"]
            return model_inputs

        train_data = datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=datasets.column_names,
            num_proc=64,
            load_from_cache_file=False,
        )

        print(f'  ==> Loaded {len(train_data)} samples')
        train_data.save_to_disk(args.cache_data)
        print(f'  ==> Saved to {args.cache_data}')
        return train_data

