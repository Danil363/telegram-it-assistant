from datasets import load_dataset
from transformers import (AutoModelForCausalLM, 
                            AutoTokenizer, 
                            Trainer, 
                            TrainingArguments,
                            DataCollatorForLanguageModeling)
from datasets import Dataset
import json
import torch

dataset = load_dataset(
    "json",
    data_files="data/conversations.jsonl",
    split="train",
    streaming=True
)

model_name = 'sberbank-ai/rugpt3medium_based_on_gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(
        example["conversation"],  
        padding="max_length",    
        truncation=True,
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize, remove_columns=["domain", "conversation"])

def add_labels(example):
    example["labels"] = example["input_ids"]
    return example

tokenized_dataset = tokenized_dataset.map(add_labels)

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

OUTPUT_DIR = "src/model_configs/rugpt-dialogue"

BATCH_SIZE = 2
EPOCHS = 3
LR = 5e-5

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,

    max_steps=50000,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,

    logging_steps=100,
    save_strategy="no", 

    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()
print("Обучение завершено!")


trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)