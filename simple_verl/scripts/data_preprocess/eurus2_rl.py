import os
import datasets

from transformers import AutoTokenizer

ds = datasets.load_dataset("PRIME-RL/Eurus-2-RL-Data", trust_remote_code=True)
train_dataset = ds['train']
test_dataset = ds['validation']

# print(set(train_dataset["data_source"]))
# print(set(test_dataset["data_source"]))

# only keep math date, discard coding
filter_fn = lambda example: example['ability'] == 'math'
train_dataset = train_dataset.filter(filter_fn)
test_dataset = test_dataset.filter(filter_fn)

new_system_prompt = """\
A conversation that the assistant solves the user's problem. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> all the reasoning process here </think> <answer> final answer here </answer>.
"""

local_dir = "../data/eurus2_rl"
os.makedirs(local_dir, mode=0o777, exist_ok=True)

def process_fn(example):
    # each data looks like:
    # data = {
    #     "data_source": ...,
    #     "prompt": [{
    #         "role": "user",
    #         "content": question
    #     }],
    #     "ability": "math",
    #     "reward_model": {
    #         "style": "rule",
    #         "ground_truth": solution
    #     },
    #     "extra_info": {
    #         'index': idx,
    #         'split': 'train',
    #     }
    # }
    for d in example['prompt']:
        if d['role'] == 'system':
            d['content'] = new_system_prompt
    return example

train_dataset = train_dataset.map(function=process_fn, num_proc=10)
test_dataset = test_dataset.map(function=process_fn, num_proc=10)

train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
q = tokenizer.apply_chat_template(train_dataset[0]['prompt'], add_generation_prompt=True, tokenize=False)
print(q)