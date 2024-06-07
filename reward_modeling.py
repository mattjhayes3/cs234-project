# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
python examples/scripts/reward_modeling.py \
    --model_name_or_path=gpt2 \
    --output_dir="models/2c2-reward" \
    --dataset="pref_pairs_16_token_2_choose_2.csv" \
    --per_device_train_batch_size=2 \
    --num_train_epochs=3 \
    --gradient_accumulation_steps=1 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to="wandb" \
    --remove_unused_columns=False \
    --optim="adamw_torch" \
    --logging_steps=10 \
    --evaluation_strategy="steps" \
    --eval_steps=100 \
    --save_steps=100
    --max_length=25 \
    --load_best_model_at_end=True \
    --save_total_limit=3 \
    --lr_scheduler_type="constant_with_warmup" \
    --warmup_steps=150
"""
import warnings

from dataclasses import dataclass, field
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, RewardConfig, RewardTrainer, get_kbit_device_map, get_peft_config, get_quantization_config

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset

tqdm.pandas()

@dataclass
class ScriptArguments:
    dataset: str = field(default=None, metadata={'help': 'dataset to use'})

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    args, config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path, num_labels=1, **model_kwargs
    )
    model.config.pad_token_id = model.config.eos_token_id

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################
    raw_dataset = Dataset.from_csv(args.dataset)
    # Tokenize chosen/rejected pairs of inputs
    # Adapt this section to your needs for custom datasets

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for prompt, chosen, rejected in zip(examples['prompt'], examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(prompt+chosen)
            tokenized_rejected = tokenizer(prompt+rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    ds = raw_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    print("len before filter", len(ds))
    ds = ds.filter(
        lambda x: len(x["input_ids_chosen"]) <= config.max_length and len(x["input_ids_rejected"]) <= config.max_length
    )
    print("len after filter", len(ds))
    ds = ds.train_test_split(test_size=0.05, shuffle=True, seed=42)
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    trainer.push_to_hub()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    print(metrics)
