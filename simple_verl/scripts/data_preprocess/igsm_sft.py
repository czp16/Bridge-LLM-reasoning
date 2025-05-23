"""
Preprocess the iGSM dataset to parquet format
"""
from typing import Dict, Any
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
    parser.add_argument('--train_dataset_path', default='../data/iGSM/SFT_op_15-20_long_cot_5K')
    parser.add_argument('--save_dir', default='../data/parquet_files/iGSM')

    args = parser.parse_args()

    train_dataset = datasets.load_from_disk(args.train_dataset_path)

    system_prompt = "A conversation that the assistant solves the user's problem. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> all the reasoning process here </think> <answer> final answer here </answer>."


    def process_fn(example: Dict[str, Any]):
        question = example.pop('query')
        answer_cot = example.pop('response')
        answer_cot_with_mask = example.pop('response_with_mask', None)
        solution = example.pop('final_answer')
        num_ops = example.pop('num_ops')

        data = {
            "data_source": 'igsm',
            "message": [{
                "role": "system",
                "content": system_prompt,
            }, {
                "role": "user",
                "content": question,
            }, {
                "role": "assistant",
                "content": answer_cot,
            }],
            "extra_info": {
                "num_ops": num_ops,
                "final_answer": solution,
                "answer_with_mask": answer_cot_with_mask,
            }
        }
        return data

    train_dataset = train_dataset.map(function=process_fn, num_proc=10)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(args.train_dataset_path) + ".parquet"
    train_dataset.to_parquet(os.path.join(save_dir, file_name))

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.chat_template = """
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
    {{- message['content'] + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""

    q = tokenizer.apply_chat_template(train_dataset[0]['message'], add_generation_prompt=False, tokenize=False)
    print(q)
