import os
import pprint
import argparse
import numpy as np
from tokenization import load_tokenize_data
import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, TrainingArguments, Trainer

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def freeze_decoder_except_xattn_codegen(model):
    print(f'Para before freezing: {model.num_parameters()}, trainable para: {get_model_size(model)}')
    for param in model.decoder.parameters():
        param.requires_grad = False

    num_decoder_layers = model.decoder.config.n_layer
    for i in range(num_decoder_layers):
        each_decoder_layer = model.decoder.transformer.h[i]
        if hasattr(each_decoder_layer, 'crossattention'):
            for param in each_decoder_layer.crossattention.parameters():
                param.requires_grad = True
            each_decoder_layer.crossattention.to(torch.float32)

        if hasattr(each_decoder_layer, 'alpha_xattn'):
            each_decoder_layer.alpha_xattn.requires_grad = True
    print(f'Para after freezing: {model.num_parameters()}, trainable para: {get_model_size(model)}')

def unfreeze_decoder(model):
    print(f'Para before freezing: {model.num_parameters()}, trainable para: {get_model_size(model)}')
    for param in model.decoder.parameters():
        param.requires_grad = True
    print(f'Para before freezing: {model.num_parameters()}, trainable para: {get_model_size(model)}')

def run_training(args, model, train_data):
    print(f"Starting main loop")

    training_args = TrainingArguments(
        report_to='tensorboard',
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        save_strategy='epoch',

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.1,
        warmup_steps=args.lr_warmup_steps,

        logging_dir=args.save_dir,
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_total_limit=2,

        dataloader_drop_last=True,
        dataloader_num_workers=2,

        local_rank=args.local_rank,
        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )

    trainer.train()
    
    trainer_history = pd.DataFrame(trainer.state.log_history[:-1]).set_index("step")
    trainer_history.loss.plot(label="loss")
    plt.ylabel("loss")
    final_loss_pdf_dir = os.path.join(args.save_dir, "loss.pdf")
    plt.savefig(final_loss_pdf_dir)
    final_loss_csv_dir = os.path.join(args.save_dir, "loss.csv")
    trainer_history.to_csv(final_loss_csv_dir)
    
    unfreeze_decoder(model)
    if args.local_rank in [0, -1]:
        final_checkpoint_dir = os.path.join(args.save_dir, "final_checkpoint_ser")
        model.save_pretrained(final_checkpoint_dir, safe_serialization=False)
        print(f'  ==> Finish training and save to {final_checkpoint_dir}')



def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # Save command to file
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    # Load and tokenize data using the tokenizer from `args.load`. If the data is already cached, load it from there.
    # You can customize this function to load your own data for any Seq2Seq LM tasks.
    train_data = load_tokenize_data(args)

    if args.data_num != -1:
        train_data = train_data.select([i for i in range(args.data_num)])

    # Load model from `args.load`
    model = AutoModelForSeq2SeqLM.from_pretrained(args.load, torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True, trust_remote_code=True)

    print(f"  ==> Loaded model from {args.load}, model size {model.num_parameters()}")
    freeze_decoder_except_xattn_codegen(model)

    run_training(args, model, train_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CodeT5+ instruction tuning")
    parser.add_argument('--data-num', default=-1, type=int)
    parser.add_argument('--prompt-type', default='general', choices=['general', 'code', 'docu-code'], type=str)
    parser.add_argument('--max-len', default=2024, type=int)
    parser.add_argument('--instruct-data-path', default='code_alpaca_20k.json', type=str)
    parser.add_argument('--cache-data', default='cache_data/instructions', type=str)
    parser.add_argument('--load', default='Salesforce/codet5p-16b', type=str)
    parser.add_argument('--with-decoder-input', default=False, action='store_true')
    parser.add_argument('--docu-encoder-input', default=False, action='store_true')

    # Training
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--lr-warmup-steps', default=30, type=int)
    parser.add_argument('--batch-size-per-replica', default=1, type=int)
    parser.add_argument('--grad-acc-steps', default=16, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--deepspeed', default=None, type=str)
    parser.add_argument('--fp16', default=False, action='store_true')

    # Logging and stuff
    parser.add_argument('--save-dir', default="saved_models/instruct_codet5p_16b", type=str)
    parser.add_argument('--log-freq', default=10, type=int)
    parser.add_argument('--save-freq', default=500, type=int)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main(args)
