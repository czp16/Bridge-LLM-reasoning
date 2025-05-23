import sys
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import argparse
import numpy as np
import json
from tqdm import tqdm
from collections import defaultdict
from vllm import LLM, SamplingParams
import torch
import re
import math
from transformers import AutoTokenizer
import datasets
from datetime import datetime

from sverl.workers.reward_manager import iGSMRewardManager

llama_chat_template = """
{%- for message in messages %}
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}
    {{- message['content'] + '<|eot_id|>' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""

def main(args):
    outputs = []

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if 'llama' in args.model_dir:
        print("Reset llama chat template.")
        tokenizer.chat_template = llama_chat_template
    
    llm = LLM(model=args.model_dir, gpu_memory_utilization=args.gpu_memory_utilization)
    
    def process_fn(example):
        return {
            "query": tokenizer.apply_chat_template(
                example["prompt"],
                add_generation_prompt=True,
                tokenize=False,
            ) + "<think>",
            "ground_truth": example["reward_model"]["ground_truth"],
        }
        
    for dataset_i in args.data_name_list:
 
        print(os.path.join(args.data_dir, dataset_i + ".parquet"))
        test_dataset = datasets.load_dataset("parquet", data_files={
            "test": os.path.join(args.data_dir, dataset_i + ".parquet"),
        })["test"]

        test_dataset = test_dataset.map(process_fn, num_proc=10)

        reward_fn = iGSMRewardManager(format_reward=0.0)
        
        if 'llama' in args.model_dir:
            stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        else: # qwen
            stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]

        sampling_params = SamplingParams(
            max_tokens=2560,
            temperature=args.temperature,
            stop_token_ids=stop_token_ids,
            # detokenize=False,
            stop=["</answer>"],
        )
        
        query_list = test_dataset["query"]
        # idx_list = test_dataset["idx"]
        ground_truth_list = test_dataset["ground_truth"]
        
        if args.num_n > 1:
            query_list = [x for x in query_list for _ in range(args.num_n)]
            # idx_list = [x for x in idx_list for _ in range(args.num_n)]
            ground_truth_list = [x for x in ground_truth_list for _ in range(args.num_n)]
        
        
        outputs = llm.generate(query_list, sampling_params=sampling_params)
        completion_ids = [output.outputs[0].token_ids for output in outputs]
        response_length_list = [len(ids) for ids in completion_ids]
        avg_length = np.mean([len(ids) for ids in completion_ids])
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=False)

        rewards = reward_fn.compute_score(completions, ground_truth_list)

        print("Average completion length:", avg_length)
        print("Accuracy:")
        print(f"{sum(rewards)} / {len(rewards)} = {sum(rewards) / len(rewards)}")
        
        accuracy_report = sum(rewards) / len(rewards)

        data = {
            "query": query_list,
            "ground_truth": ground_truth_list,
            "completion": completions,
            "reward": rewards,
            "response_length": response_length_list,
            # "idx": idx_list
        }
        # dict of list to list of dict
        list_of_dict = [dict(zip(data, t)) for t in zip(*data.values())]
        
        save_dir = os.path.join(args.model_dir, args.save_dir)
        date_str = datetime.now().strftime('%Y-%m-%d')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset_i}-tem_{args.temperature}-num_n-{args.num_n}-{date_str}.json")
        with open(save_path, "w") as f:
            json.dump(list_of_dict, f, indent=2)
        acc_save_path = os.path.join(save_dir, f"{dataset_i}-acc.txt")
        with open(acc_save_path, "w") as f:
            f.write(f"{accuracy_report:.4f}\n")  # or str(value)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/parquet_files/iGSM")
    parser.add_argument(
        '--data_name_list',
        nargs='+',
        type=str,
        default=[
            # "test_op_10_500", 
            # "test_op_15_500",
            "test_500_no_CoT_op_20",
            "test_500_no_CoT_op_25",
            # "test_op_30_500",
        ],
        help='List of dataset names to be evaluated'
    )
    
    parser.add_argument("--model_dir", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="generated_response")
    parser.add_argument("--num_n", type=int, default=1)  # number of N for each question
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    
    args = parser.parse_args()
    main(args)