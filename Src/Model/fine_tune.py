from datetime import datetime
import argparse
import torch
from peft import (
    LoraConfig,
    get_peft_model
)
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import datasets
import random

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

#############Parse Arg#############
parser = argparse.ArgumentParser(description='Fine Tune')
parser.add_argument('-m','--model')
parser.add_argument('-token-num','--token_num')
parser.add_argument('-lr','--lr')
parser.add_argument('-dataset-dir', '--dataset_dir')
parser.add_argument('-epochs', '--epochs')
parser.add_argument('-output-dir', '--output_dir')
parser.add_argument('-adapters-dir', '--adapters_dir')


args = parser.parse_args()
base_model = args.model
token_num = args.token_num
lr = args.lr
dataset_dir = args.dataset_dir
num_train_epochs = args.epochs
output_dir = args.output_dir
adapters_dir = args.adapters_dir

#############Tokenize Dataset#############
train_dataset = datasets.load_from_disk(f"{dataset_dir}/train_dataset")
eval_dataset = datasets.load_from_disk(f"{dataset_dir}/eval_dataset")

wandb_project = f"{base_model}-fine-tune"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.add_eos_token = True
tokenizer.pad_token_id = 2
tokenizer.padding_side = "left"

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

#############Fine Tune#############
model.train() # put model back into training mode

if "codellama" in base_model:
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
elif "codegemma" in base_model:
    config = LoraConfig(
        r=8,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

model = get_peft_model(model, config)

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

if adapters_dir == None:
    trainer.train()
else:
    print(f"Load from {adapters_dir}...")
    trainer.train(resume_from_checkpoint=adapters_dir)

print("train done")
