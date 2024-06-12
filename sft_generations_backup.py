from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

import torch.nn.functional as F

model_name = "lvwerra/gpt2-imdb"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

device = 0 if torch.cuda.is_available() else "cpu"


def build_dataset(dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
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

dataset = build_dataset()


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collator,
    shuffle=False,
    drop_last=False,
)

tokenizer.pad_token = tokenizer.eos_token
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 32,
    # "num_return_sequences": 4
}

config = PPOConfig(
    model_name="lvwerra/gpt2-imdb",
    # model_name="edbeeching/gpt2-large-imdb",
    learning_rate=1.41e-5,
    # log_with="wandb",
    # init_kl_coef=0.02,
    init_kl_coef=1.0,
    # target=10,
    adap_kl_ctrl=False,
    # steps=1000,
    batch_size=128,
    mini_batch_size=128,
    # reward_model="sentiment-analysis:lvwerra/distilbert-imdb",
    reward_model="sentiment-analysis:siebert/sentiment-roberta-large-english",
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ppo_trainer = PPOTrainer(config, model, model, tokenizer, dataset=dataset, data_collator=collator)

sent_kwargs = {"return_all_scores": True, "function_to_apply": "softmax", "batch_size": 32}
task, model_name = config.reward_model.split(":")
print("Load reward model", model_name)
sentiment_pipe = pipeline(task, model=model_name, device=device)

# Some tokenizers like GPT-2's don't have a padding token by default, so we set one here.
if sentiment_pipe.tokenizer.pad_token_id is None:
    sentiment_pipe.tokenizer.pad_token_id = tokenizer.pad_token_id

all_results = []
parts = 0
with torch.no_grad():
  for _epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
      if len(all_results) * config.batch_size > 5000:
        pd.concat(all_results, axis=0).reset_index(drop=True).to_csv(f'sft_generations_{parts}.csv')
        parts += 1
        all_results = []
      # if  _epoch > 1:
      #     break
      query_tensors = batch["input_ids"]

      batch_results = {'query': batch['query']}
      for gen_num in range(4):
        print('gen num', gen_num)
        # Get response from gpt2
        response_tensors= ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=False, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        # batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        print("texts", texts)
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        batch_results[f'generation_{gen_num}'] = batch['response']
        batch_results[f'sentiment_{gen_num}'] = [output[1]["score"] for output in pipe_outputs]
      batch_df = pd.DataFrame(batch_results)
      # print(batch_df)
      all_results.append(batch_df)
pd.concat(all_results, axis=0).reset_index(drop=True).to_csv(f'sft_generations_{parts}.csv')

