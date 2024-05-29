# flake8: noqa
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
# regular:
python examples/scripts/dpo.py \
    --dataset_name=DownwardSpiral33/gpt2-imdb-16-token-pref-pairs \
    --model_name_or_path=lvwerra/gpt2-imdb \
    --per_device_train_batch_size 256 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --warmup_steps 150 \
    --report_to wandb \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

import logging
import multiprocessing
import os
from contextlib import nullcontext
import ppo

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser

if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


if TRL_USE_RICH:
    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO)


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()
    full_name = f"{training_args.output_dir}-{training_args.start_time}"
    training_args.output_dir = full_name

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

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
        # attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    print(f"Load model ", model_config.model_name_or_path)
    peft_config = get_peft_config(model_config)
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)
        print(f"Load ref model ", model_config.model_name_or_path)
    else:
        model_ref = None
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # if tokenizer.chat_template is None:
    #     tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Optional rich context managers
    ###############
    init_context = nullcontext() if not TRL_USE_RICH else console.status("[bold green]Initializing the DPOTrainer...")
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(f"[bold green]Training completed! Saving the model to {training_args.output_dir}")
    )

    ################
    # Dataset
    ################
    ds = load_from_disk(args.dataset_name) if not args.tokenize else load_dataset(args.dataset_name)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    def process(row):
        # row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        # row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    if not args.tokenize:
        if isinstance(ds, DatasetDict):
            ds = ds['train']
        print('ds len', len(ds))
        ds = ds.train_test_split(test_size=0.05, shuffle=True, seed=42)
        # print(len(ds), type(ds))
        # ds.map(
        #     process,
        #     num_proc=multiprocessing.cpu_count(),
        #     load_from_cache_file=False,
        # )
        train_dataset = ds[args.dataset_train_split]
        eval_dataset = ds[args.dataset_test_split]

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=ds if args.tokenize else train_dataset,
            eval_dataset=None if args.tokenize else eval_dataset,
            tokenizer=tokenizer,
            peft_config=get_peft_config(model_config),
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
            tokenize=args.tokenize,
        )

    if args.tokenize:
        trainer.train_dataset.save_to_disk(f"{args.dataset_name}_tokenized", max_shard_size='90MB')
    else:
        trainer.train()
        with save_context:
            trainer.save_model(training_args.output_dir)
        steps = len(trainer.get_train_dataloader())
        print(f"Evaling every {steps}")
        results = []
        for epoch in range(1, training_args.num_train_epochs+1):
            r, r_sem, kl, kl_sem = ppo.eval(f"{training_args.output_dir}/checkpoint-{epoch*steps}", f"epoch {epoch}")
            results.append([training_args.output_dir, training_args.beta, f"epoch {epoch}"], r, r_sem, kl, kl_sem)
        for result in results:
            print(",".join(result))