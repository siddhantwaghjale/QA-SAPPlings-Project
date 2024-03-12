import os
import sys
import glob
import transformers
import argparse
import torch
from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets, interleave_datasets
from transformers import AutoModelForSeq2SeqLM, AutoConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoTokenizer
from transformers.optimization import AdamW
from transformers import set_seed
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from multitabqa_processor import MultiTabQAProcessor
from utils import *

os.environ["WANDB_DISABLED"] = "true"
os.environ['TOKENIZERS_PARALLELISM'] = "false"

arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, help="name of dataset to adapter-tune on")
parser.add_argument("--max_length", default=1024, type=int, help="encoder sequence max length")
parser.add_argument("--decoder_max_length", default=1024, type=int, help="decoder sequence max length")
parser.add_argument("--pretrained_model_name", type=str, default=None, help="prtrained model name")
parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--lr_scheduler", default="polynomial", choices=arg_to_scheduler_choices,
                    metavar=arg_to_scheduler_metavar, type=str, help="Learning rate scheduler", )
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for AdamW optimizer.")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight Decay for AdamW optimizer.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--max_grad_norm", default=0.1, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
parser.add_argument("--use_multiprocessing", default=True, type=bool, help="use multiple processes for data loading")
parser.add_argument("--num_train_epochs", default=30, type=int)
parser.add_argument("--train_batch_size", default=4, type=int)
parser.add_argument("--eval_batch_size", default=4, type=int)
parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
parser.add_argument("--eval_gradient_accumulation", default=8, type=int)
parser.add_argument("--adafactor", action="store_true")
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--cpu", action="store_true", help="train using cpu")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="resume training from a checkpoint")
parser.add_argument("--gradient_checkpointing", action="store_true")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--table_type", type=str, default='html')



args = parser.parse_args()
use_cuda = not args.cpu
device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
seed = args.seed

print("Device: ", device)

def model_init():
    set_seed(args.seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_name)
    model.config.max_length = args.decoder_max_length
    model = model.to(device)
    return model


tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name)
config = AutoConfig.from_pretrained(args.pretrained_model_name)

def get_dataset(dataset_name):
    # pre-training stage 2
    if dataset_name in "spider_sql":
        raise Exception("Not implemented")

    # spider natural language question fine-tuning dataset
    elif dataset_name == "spider_nq":
        ds = load_dataset("kpriyanshu256/MultiTabQA-spider_nq", streaming=not True)
        train_set = ds['train']
        valid_set = ds['validation']
        if 'test' in ds:
            test_set = ds['test']
        else:
            test_set = None
        
        print(f"Training with {len(train_set)} samples, evaluating with {len(valid_set)} samples")

    # atis natural language question finetuning dataset
    elif dataset_name == "atis":
        ds = load_dataset("kpriyanshu256/MultiTabQA-atis", streaming=not True)
        train_set = ds['train']
        valid_set = ds['validation']
        test_set = ds['test']
        
    # geoquery natural language question finetuning dataset
    elif dataset_name == "geoquery":
        ds = load_dataset("kpriyanshu256/MultiTabQA-geoquery", streaming=not True)
        train_set = ds['train']
        valid_set = ds['validation']
        test_set = ds['test']
        

    # print(f"Training with {len(train_set)} samples, evaluating with {len(valid_set)} samples")

    processor = MultiTabQAProcessor(training_dataset=train_set, eval_dataset=valid_set, tokenizer=tokenizer,
                                    decoder_start_token_id=config.decoder_start_token_id,
                                    decoder_max_length=args.decoder_max_length)   
       
    return train_set, valid_set, test_set, processor


def tokenize_sample(sample):
    input_encoding = tokenizer(sample[f'source_{args.table_type}'].strip().lower().replace('"', ''),
                               return_tensors="pt",
                               padding='max_length',
                               max_length=1024,
                               truncation='longest_first',
                               add_special_tokens=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text=sample[f'target_{args.table_type}'].strip().lower().replace('"', ''),
            add_special_tokens=True,
            return_tensors="pt",
            padding='max_length',
            max_length=1024,
            truncation='longest_first',
        )
    
    return {
        "input_ids": input_encoding["input_ids"],
        "attention_mask": input_encoding["attention_mask"],
        "labels": labels['input_ids'],
    }

def tokenize_batch(samples):
    input_encoding = tokenizer([x.strip().lower().replace('"', '') for x in samples[f'source_{args.table_type}']],
                               return_tensors="pt",
                               padding='max_length',
                               max_length=args.max_length,
                               truncation='longest_first',
                               add_special_tokens=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            text=[x.strip().lower().replace('"', '') for x in samples[f'target_{args.table_type}']],
            add_special_tokens=True,
            return_tensors="pt",
            padding='max_length',
            max_length=args.decoder_max_length,
            truncation='longest_first',
        )
    
    return {
        "input_ids": input_encoding["input_ids"],
        "attention_mask": input_encoding["attention_mask"],
        "labels": labels['input_ids'],
    }



train_dataset, valid_dataset, test_dataset, processor = get_dataset(args.dataset_name)
print("#############Data loading done!#############")

print(train_dataset)
print(valid_dataset)
print(test_dataset)



train_dataset = train_dataset.map(tokenize_sample, 
            remove_columns=['query', 'table_names', 'tables', 'answer', 'source', 'target', 'source_latex', 'target_latex', 'source_html', 'target_html', 'source_markdown', 'target_markdown'])
valid_dataset = valid_dataset.map(tokenize_sample,
            remove_columns=['query', 'table_names', 'tables', 'answer', 'source', 'target', 'source_latex', 'target_latex', 'source_html', 'target_html', 'source_markdown', 'target_markdown'])

if test_dataset:
    test_dataset = test_dataset.map(tokenize_sample,
            remove_columns=['query', 'table_names', 'tables', 'answer', 'source', 'target', 'source_latex', 'target_latex', 'source_html', 'target_html', 'source_markdown', 'target_markdown'])
    


def em_metric_builder(tokenizer):
    def compute_em_metrics(pred):
        """utility to compute Exact Match during training."""
        # All special tokens are removed.
        pred_ids, labels_ids = pred.predictions, pred.label_ids
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        em = load_metric("exact_match")
        scores = em.compute(predictions=pred_str, references=label_str, ignore_case=True)
        print(f"Exact Match Scores: {scores}")
        return {
            "exact_match": round(scores['exact_match'], 4),
        }

    return compute_em_metrics


em_metric_fn = em_metric_builder(tokenizer)

train_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    do_train=True,
    do_eval=True,
    evaluation_strategy="epoch",
    no_cuda=args.cpu,
    fp16=True if use_cuda else False,
    save_strategy="epoch",
    save_total_limit=1,
    logging_steps=500,
    eval_accumulation_steps=args.eval_gradient_accumulation,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    num_train_epochs=args.num_train_epochs,
    seed=seed,
    disable_tqdm=False,
    predict_with_generate=True,
    generation_max_length=args.decoder_max_length,
    generation_num_beams=4,
    load_best_model_at_end=True,
    remove_unused_columns=not False,
    dataloader_num_workers=args.num_workers,
    metric_for_best_model="exact_match",
    dataloader_drop_last=True,
    adam_epsilon=args.adam_epsilon,
    weight_decay=args.weight_decay,
    max_grad_norm=args.max_grad_norm,
    lr_scheduler_type=args.lr_scheduler,
    warmup_steps=args.warmup_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    local_rank=args.local_rank,
)

transformers.logging.set_verbosity_info()
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=processor.collate_tokenized,
    compute_metrics=em_metric_fn,
)

print("Starting Training...")
if args.resume_from_checkpoint:
    print("Resuming training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
else:
    trainer.train()

trainer.save_state()
trainer.save_model(args.output_dir)

print("Eval metrics")
print(trainer.evaluate(valid_dataset))

if test_dataset:
    print("Test metrics")
    print(trainer.evaluate(test_dataset))
