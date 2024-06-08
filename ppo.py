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
python examples/scripts/ppo.py \
    --log_with=wandb
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, pipeline

from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from trl.import_utils import is_npu_available, is_xpu_available
import pandas as pd
import os

tqdm.pandas()

@dataclass
class ScriptArguments:
    use_seq2seq: bool = field(default=False, metadata={"help": "whether to use seq2seq"})
    trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})

    # LoraConfig
    use_peft: bool = field(default=False, metadata={"help": "whether to use peft"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_r: Optional[int] = field(default=16, metadata={"help": "the lora r parameter"})

def run(ppo_config, args, full_name):
    is_custom_reward = 'DownwardSpiral33' in ppo_config.reward_model
    reward_index = 0 if is_custom_reward else 1
    # We then define the arguments to pass to the sentiment analysis pipeline.
    # We set `return_all_scores` to True to get the sentiment score for each token.
    sent_kwargs = {"return_all_scores": True, "batch_size": 32}
    if not is_custom_reward:
        sent_kwargs.update({"function_to_apply": "softmax"})


    trl_model_class = AutoModelForCausalLMWithValueHead if not args.use_seq2seq else AutoModelForSeq2SeqLMWithValueHead


    # Below is an example function to build the dataset. In our case, we use the IMDB dataset
    # from the `datasets` library. One should customize this function to train the model on
    # its own dataset.
    def build_dataset(config, query_dataset, split, input_min_text_length=2, input_max_text_length=8):
        """
        Build dataset for training. This builds the dataset from `load_dataset`, one should
        customize this function to train the model on its own dataset.

        Args:
            query_dataset (`str`):
                The name of the dataset to be loaded.

        Returns:
            dataloader (`torch.utils.data.DataLoader`):
                The dataloader for the dataset.
        """
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        # load imdb with datasets
        ds = load_dataset(query_dataset, split=split)
        ds = ds.rename_columns({"text": "review"})
        ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

        input_size = LengthSampler(input_min_text_length, input_max_text_length)

        def tokenize(sample):
            sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
            sample["query"] = tokenizer.decode(sample["input_ids"])
            return sample

        ds = ds.map(tokenize, batched=False)
        ds.set_format(type="torch")
        return ds


    # We retrieve the dataloader by calling the `build_dataset` function.
    dataset = build_dataset(ppo_config, ppo_config.query_dataset, "train" if not ppo_config.eval_model else "train[:1%]")


    def collator(data):
        return {key: [d[key] for d in data] for key in data[0]}


    # set seed before initializing value head for deterministic eval
    set_seed(ppo_config.seed)

    # Now let's build the model, the reference model, and the tokenizer.
    if not args.use_peft:
        print(f"load ref model", ppo_config.model_name)
        ref_model = trl_model_class.from_pretrained(ppo_config.model_name, trust_remote_code=args.trust_remote_code)
        device_map = None
        peft_config = None
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
        ref_model = None
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}

    train_model_name = ppo_config.eval_model if ppo_config.eval_model else ppo_config.model_name
    print(f"load train model", train_model_name)
    model = trl_model_class.from_pretrained(
        train_model_name,
        trust_remote_code=args.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config,
    )

    tokenizer = AutoTokenizer.from_pretrained(ppo_config.model_name)

    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # We then build the PPOTrainer, passing the model, the reference model, the tokenizer
    ppo_trainer = PPOTrainer(ppo_config, 
                             model, 
                             ref_model, 
                             tokenizer, 
                             dataset=dataset, 
                             data_collator=collator)

    score_shift = 0 if not ppo_config.normalize_scores else 0.5
    score_scale = 1 if not ppo_config.normalize_scores else 2

    # We then build the sentiment analysis pipeline, passing the model name and the
    # sentiment analysis pipeline arguments. Let's also make sure to set the device
    # to the same device as the PPOTrainer.
    device = ppo_trainer.accelerator.device
    print('device', device)
    if ppo_trainer.accelerator.num_processes == 1:
        if is_xpu_available():
            device = "xpu:0"
        elif is_npu_available():
            device = "npu:0"
        # elif torch.backends.mps.is_available():
        #     device = 'mps'
        else:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    print('device', device)
    ds_plugin = ppo_trainer.accelerator.state.deepspeed_plugin
    task, model_name = ppo_config.reward_model.split(":")
    print("Load reward model", model_name)
    if ds_plugin is not None and ds_plugin.is_zero3_init_enabled():
        with ds_plugin.zero3_init_context_manager(enable=False):
            sentiment_pipe = pipeline(task, model=model_name, device=device)
    else:
        sentiment_pipe = pipeline(task, model=model_name, device=device)

    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    if sentiment_pipe.tokenizer.pad_token_id is None:
        sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

    if sentiment_pipe.model.config.pad_token_id is None:
        sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id

    # We then define the arguments to pass to the `generate` function. These arguments
    # are passed to the `generate` function of the PPOTrainer, which is a wrapper around
    # the `generate` function of the trained model.
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": ppo_config.max_new_tokens,
    }

    if not ppo_config.eval_model:
        for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
            if ppo_config.dry_run and _epoch > 0:
                break
            query_tensors = batch["input_ids"]

            # Get response from gpt2
            response_tensors, ref_response_tensors = ppo_trainer.generate(
                query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
            )
            batch["response"] = tokenizer.batch_decode(response_tensors)
            batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

            # Compute sentiment score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
            # print(pipe_outputs)
            rewards = [(torch.tensor(output[reward_index]["score"]) - score_shift) * score_scale for output in pipe_outputs]
            ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
            ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
            ref_rewards = [(torch.tensor(output[reward_index]["score"]) - score_shift) * score_scale for output in ref_pipe_outputs]
            batch["ref_rewards"] = ref_rewards

            # Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            if not ppo_config.ac2:
                ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

        if not ppo_config.dry_run:
            model.save_pretrained(f"models/gpt2-imdb-pos-{full_name}", push_to_hub=True)
            tokenizer.save_pretrained(f"models/gpt2-imdb-pos-{full_name}", push_to_hub=True)
        print("Training done!  Start eval")


    print("Load eval reward model sentiment-analysis:siebert/sentiment-roberta-large-english")
    sentiment_pipe = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english', device=device)

    # Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
    if sentiment_pipe.tokenizer.pad_token_id is None:
        sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

    if sentiment_pipe.model.config.pad_token_id is None:
        sentiment_pipe.model.config.pad_token_id = tokenizer.pad_token_id
    sent_kwargs = {"return_all_scores": True, "batch_size": 32, "function_to_apply": "softmax"}

    print("eval batch size", ppo_trainer.config.batch_size)
    dataset = build_dataset(ppo_trainer.config, ppo_config.query_dataset, "test[:10%]")#[:512]
    dataloader = ppo_trainer.prepare_dataloader(dataset, collator)
    print("test len", len(dataloader))
    test_stats = []
    for _epoch, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        if ppo_config.dry_run and _epoch > 0:
            break
        query_tensors = batch["input_ids"]

        # Get response from gpt2
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)
        batch['ref_length'] = [len(tokenizer.encode(r)) for r in batch['ref_response']]
        batch['length'] = [len(tokenizer.encode(r)) for r in batch['response']]

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        # print(pipe_outputs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        batch["rewards"] = rewards
        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
        ref_rewards = [output[1]["score"] for output in ref_pipe_outputs]
        batch["ref_rewards"] = ref_rewards

        # Run PPO step
        batch["full_kls"] = ppo_trainer.get_generation_kls("full", query_tensors, response_tensors, rewards).sum(dim=-1).cpu().tolist()
        batch["kls"] = ppo_trainer.get_generation_kls("kl", query_tensors, response_tensors, rewards).sum(dim=-1).cpu().tolist()
        batch["rewards"] = [reward.item() for reward in batch["rewards"]]
        # print(f"kls shape", batch["kls"].shape)
        # print(f"full kls shape", batch["full_kls"].shape)
        batch_df = pd.DataFrame(batch)
        # print(batch_df)
        test_stats.append(batch_df[['query', 'response', 'rewards', 'ref_response', 'ref_rewards', 'full_kls', 'kls', 'ref_length', 'length']])


    test_stats = pd.concat(test_stats, axis=0).reset_index(drop=True)
    path = f'./results/{full_name}.csv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    test_stats.to_csv(path)
    print("mean test reward", test_stats.rewards.mean(), "+/-", test_stats.rewards.sem(), "from", test_stats.ref_rewards.mean(), "+/-", test_stats.ref_rewards.sem())
    print("median test reward", test_stats.rewards.median(), "from", test_stats.ref_rewards.median())
    print("mean KL", test_stats.kls.mean(), "+/-", test_stats.kls.sem(), "full", test_stats.full_kls.mean(), "+/-", test_stats.full_kls.sem())
    print("median KL", test_stats.kls.median(), "full", test_stats.full_kls.median())
    return test_stats.rewards.mean(), test_stats.rewards.sem(), test_stats.full_kls.mean(), test_stats.full_kls.sem(), test_stats.length.mean()

def eval(model, notes):
    return run(PPOConfig(exp_name="eval", eval_model=model), args=ScriptArguments(), full_name=f'{model}_{notes}')

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig))
    args, ppo_config = parser.parse_args_into_dataclasses()
    print("PPOConfig:", ppo_config)
    if ppo_config.eval_model:
        full_name = ppo_config.eval_model
    else:
        full_name = f"{ppo_config.exp_name}-{ppo_config.start_time}"
    stat = [full_name, ppo_config.init_kl_coef, "epoch 1", *run(ppo_config, args, full_name)]
    toplevel = ",".join([str(s) for s in stat])
    print(toplevel)
    with open('results/toplevel.csv', 'a') as f:
        f.write(toplevel + '\n')
