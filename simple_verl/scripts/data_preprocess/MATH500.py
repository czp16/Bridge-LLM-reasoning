import os
import datasets

from transformers import AutoTokenizer

ds = datasets.load_dataset("HuggingFaceH4/MATH-500")
test_dataset = ds['test']

new_system_prompt = """\
A conversation that the assistant solves the user's problem. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> all the reasoning process here </think> <answer> final answer here </answer>.
"""

local_dir = "/mnt/newhome/zhepeng/projects/data/MATH500"
os.makedirs(local_dir, mode=0o777, exist_ok=True)

def process_fn(example):
    # each data looks like:
    # data = {
    #     "prompt": [{
    #         "role": "user",
    #         "content": question
    #     }],
    #     "reward_model": {
    #         "style": "rule",
    #         "ground_truth": solution
    #     },
    # }
    d = {
        "prompt": [{
            "role": "system",
            "content": new_system_prompt
        },{
            "role": "user",
            "content": example['problem']
        }],
        "reward_model": {
            "style": "rule",
            "ground_truth": example['answer']
        },
    }
    return d

test_dataset = test_dataset.map(function=process_fn, num_proc=10)
test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
q = tokenizer.apply_chat_template(test_dataset[0]['prompt'], add_generation_prompt=True, tokenize=False)
print(q)