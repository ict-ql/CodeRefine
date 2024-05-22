from datetime import datetime
from logging import root
import os
import sys
from peft import PeftModel

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from utils.custom_data_load import load_dataset
from transformers import T5Config, T5ForConditionalGeneration, PreTrainedTokenizerFast
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import datasets
import random

os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# export all_proxy=socks5://127.0.0.1:7890

token_num = 2500
# lr = 1e-4
lr = 1e-4

date = "0406"

# fine_tune_label = "vec-phi2Optbb"
# fine_tune_label = "exebench-full-unopt2Optbb"
# fine_tune_label = "exebench-full-unopt2optCFG"
fine_tune_label = "exebench-full-unopt2Optbb-no-retriever"
# fine_tune_label = "exebench-small-phi2Optbb"
# fine_tune_label = "exebench-small-unopt2Opt"
# fine_tune_label = "exebench-small-unopt2Optbb"
# fine_tune_label = "exebench-small-mir-ir-func"
# fine_tune_label = "exebench-small-unopt2Opt"
root_dir = f"/home/ql/SRC/ActionModel/Dataset/{fine_tune_label}/{fine_tune_label}-{date}"

train_dataset = datasets.load_from_disk(f"{root_dir}/train_dataset")
eval_dataset = datasets.load_from_disk(f"{root_dir}/eval_dataset")
sample_eval_dataset_path = f"{root_dir}/sample_eval_dataset"

# if kind == "pretrain-cfg" or kind == "pretrain-loop":
sample_num = 10000
if os.path.exists(sample_eval_dataset_path):
    print(f'Load from `{sample_eval_dataset_path}`!')
    eval_dataset = datasets.load_from_disk(sample_eval_dataset_path)
else:
    if len(eval_dataset) > sample_num:
        indices = random.sample(range(len(eval_dataset)), k=sample_num)
        eval_dataset = eval_dataset.select(indices) #sample
        print(f"sample eval_dataset to `{len(eval_dataset)}`!")
        eval_dataset.save_to_disk(sample_eval_dataset_path)

output_dir = f"Adapters/{fine_tune_label}/codellama-7b-adapters-{fine_tune_label}-{date}"
adapters_dir = f"Adapters/{fine_tune_label}/codellama-7b-adapters-{fine_tune_label}-{date}/checkpoint-1100000"
# base_model = "codellama/CodeLlama-13b-Instruct-hf"
base_model = "codellama/CodeLlama-7b-Instruct-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)


num_train_epochs = 100

# finetuning script
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 2
tokenizer.padding_side = "left"
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=token_num,
        padding=False,
        return_tensors=None,
    )

    # "self-supervised learning" means the labels are also the inputs:
    result["labels"] = result["input_ids"].copy()

    return result
def generate_and_tokenize_prompt(data_point):
    text = data_point["text"]
    full_prompt =f"""{text}
"""
    return tokenize(full_prompt)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

model.train() # put model back into training mode

config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

# if not os.path.exists(adapters_dir):
#     model = get_peft_model(model, config) #@TODO:换成如果存在xxx文件，就 model = PeftModel.from_pretrained(model, adapter_name)，否则现在这样
# else:
#     model = PeftModel.from_pretrained(model, adapters_dir)
#     print(f"load from {adapters_dir}...")

# wandb_project = "codellama-13b-unopt2opt"
wandb_project = f"codellama-7b-unopt2opt-exebench-{fine_tune_label}"
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_API_KEY"] = "55997e247b7e3c2d8e6ff2314d2cd9c8e2158706"
    os.environ["WANDB_MODE"] = "offline"

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

batch_size = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = batch_size // per_device_train_batch_size


training_args = TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs = num_train_epochs,
        warmup_steps=100,
        learning_rate=lr,
        fp16=True,
        logging_steps=100,
        optim="adamw_torch",
        evaluation_strategy="steps", # if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=50000,
        save_steps=50000,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True,
        # ddp_find_unused_parameters=False if ddp else None,
        group_by_length=True, # group sequences of roughly the same length together to speed up training
        report_to="wandb", # if use_wandb else "none",
        run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}", # if use_wandb else None,
    )

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)

model.config.use_cache = False

# old_state_dict = model.state_dict
# model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
#     model, type(model)
# )
# print("compiling the model")
# model = torch.compile(model)

if not os.path.exists(adapters_dir):
    trainer.train()
else:
    print(f"Load from {adapters_dir}...")
    trainer.train(resume_from_checkpoint=adapters_dir)
# output_dir = os.path.join(output_dir, "final_checkpoint")
# trainer.model.save_pretrained(output_dir)
print("train done")
