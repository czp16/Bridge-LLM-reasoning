"""
Preprocess the iGSM dataset to parquet format
"""

import re
import os
import datasets

import argparse
from transformers import AutoTokenizer


# def extract_solution(solution_str):
#     solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
#     assert solution is not None
#     final_solution = solution.group(0)
#     final_solution = final_solution.split('#### ')[1].replace(',', '')
#     return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset_path', default='../data/iGSM/RL_train_op_15-20_long_cot_50K')
    parser.add_argument('--val_dataset_path', default='../data/iGSM/RL_val_op_15-20_long_cot_500')
    parser.add_argument('--save_dir', default='../data/parquet_files/iGSM')

    args = parser.parse_args()

    train_dataset = datasets.load_from_disk(args.train_dataset_path)
    val_dataset = datasets.load_from_disk(args.val_dataset_path)

    system_prompt = "A conversation that the assistant solves the user's problem. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> all the reasoning process here </think> <answer> final answer here </answer>."


    def process_fn(example):
        question = example.pop('query')
        answer_cot = example.pop('response')
        solution = example.pop('final_answer')
        num_ops = example.pop('num_ops')

        data = {
            "data_source": 'igsm',
            "prompt": [{
                "role": "system",
                "content": system_prompt,
            }, {
                "role": "user",
                "content": question,
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                "num_ops": num_ops,
                "answer": answer_cot,
            }
        }
        return data

    train_dataset = train_dataset.map(function=process_fn, num_proc=10)
    val_dataset = val_dataset.map(function=process_fn, num_proc=10)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    train_dataset_file_name = os.path.basename(args.train_dataset_path) + ".parquet"
    val_dataset_file_name = os.path.basename(args.val_dataset_path) + ".parquet"

    train_dataset.to_parquet(os.path.join(save_dir, train_dataset_file_name))
    val_dataset.to_parquet(os.path.join(save_dir, val_dataset_file_name))

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
    q = tokenizer.apply_chat_template(train_dataset[0]['prompt'], add_generation_prompt=True, tokenize=False)
    print(q)
