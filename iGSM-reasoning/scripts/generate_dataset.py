import os
import random
import numpy as np
import datasets
import json
import argparse
from igsm_reasoning.generation import ProblemGenerator
from igsm_reasoning.utils.misc import _strip_markers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_operations', type=int, default=20)
    parser.add_argument('--min_operations', type=int, default=15)
    parser.add_argument('--dataset_size', type=int, default=-1)
    parser.add_argument('--force', action='store_true', default=False, help='if true, all op=max_op')
    parser.add_argument('--debug', action='store_true', default=False)

    parser.add_argument('--reflection_prob', type=float, default=0.1)
    parser.add_argument('--analysis_level', type=int, default=2, choices=[0,1,2,3,4], help='probability of analysis')
    # parser.add_argument('--shuffle_query', action='store_true', default=False, help='if true, shuffle the query')
    # parser.add_argument('--redundant_query', action='store_true', default=False, help='if true, use redundant query')
    parser.add_argument('--detailed_computation', action='store_true', default=False, help='if true, use detailed computation')
    parser.add_argument('--content_mask', action='store_true', default=False, help='if true, use content mask')
    parser.add_argument('--question_aug', type=int, default=-1, help='number of augmentations for each question')
    parser.add_argument('--answer_aug', type=int, default=-1, help='number of augmentations for each answer')
    
    parser.add_argument('--use_cot', action='store_true', default=False, help='if true, use long CoT')
    parser.add_argument('--max_solution', type=int, default=1000, help='max number of solutions')
    parser.add_argument('--data_type', type=str, choices=['SFT', 'RL_train', 'RL_val', 'test'])
    parser.add_argument('--save_dir', type=str, default='data/iGSM')

    args = parser.parse_args()

    config = {
        # "english_path": "igsm_reasoning/generation/english/categorization.json",
        "max_operations": 20,
        "min_operations": 15,
        "force": False,
        "calculate_mod": False, # do not calculate the mod value
        "rng_range": 10,
        "debug": False,

        "reflection_prob": 0.1,
        "analysis_prob_list": [0, 0.1, 0.4, 0.8, 1, 1, 1, 1, 1, 1, 1],
        "shuffle_query": True,
        "redundant_query": True,
        "detailed_computation": False,
        "content_mask": False,
    }

    for k, v in vars(args).items():
        if k in config:
            config[k] = v

    if args.analysis_level == 0:
        config["analysis_prob_list"] = [0 for _ in range(len(config["analysis_prob_list"]))]
    elif args.analysis_level == 1:
        config["analysis_prob_list"] = [0, 0, 0.1, 0.4, 0.8, 1, 1, 1, 1, 1, 1]
    elif args.analysis_level == 2:
        config["analysis_prob_list"] = [0, 0.1, 0.4, 0.8, 1, 1, 1, 1, 1, 1, 1]
    elif args.analysis_level == 3:
        config["analysis_prob_list"] = [0, 0.4, 0.8, 1, 1, 1, 1, 1, 1, 1, 1]
    elif args.analysis_level == 4:
        config["analysis_prob_list"] = [0] + [1 for _ in range(len(config["analysis_prob_list"]) - 1)]
    

    if config["force"]:
        op_num = str(config['max_operations'])
    else:
        op_num = f"{config['min_operations']}-{config['max_operations']}"

    if args.answer_aug > 0:
        assert args.use_cot, "answer augmentation only works with CoT"
    assert args.answer_aug < 0 or args.question_aug < 0

    SEEDS = {
        "SFT": 100,
        "RL_train": 200,
        "RL_val": 300,
        "test": 400,
    }

    DATASET_SIZES = {
        "SFT": 5000,
        "RL_train": 10000,
        "RL_val": 500,
        "test": 500,
    }

    data_type = args.data_type
    SEED = SEEDS[data_type]
    if args.dataset_size > 0:
        dataset_size = args.dataset_size
    else:
        dataset_size = DATASET_SIZES[data_type]
    config["dataset_size"] = dataset_size

    ds_size_str = f"{dataset_size//1000}K" if dataset_size >= 1000 else str(dataset_size)
    detailed_computation_str = "detailed" if args.detailed_computation else "no_detailed"
    if args.question_aug > 0:
        other_desc = f"_{args.question_aug}Qaug"
    elif args.answer_aug > 0:
        other_desc = f"_{args.answer_aug}Aaug"
    else:
        other_desc = ""
    if args.use_cot:
        file_name = f"{data_type}_{ds_size_str}{other_desc}_CoT_op_{op_num}_{detailed_computation_str}_reflect_{args.reflection_prob}_analysis_{args.analysis_level}"
    else:
        file_name = f"{data_type}_{ds_size_str}_no_CoT_op_{op_num}"
    save_path = os.path.join(args.save_dir, file_name)
    print(f"Saving to {save_path}")

    pg = ProblemGenerator(config, seed=SEED)

    questions = []
    answers = []
    answers_with_mask = []
    num_ops = []
    final_answers = []

    # we want to reduce the number of zero answers, because it can cause false positives
    # here it only works for validation and test sets
    reduce_zero_answer = (args.min_operations > 20)
    reduce_zero_ratio = 0.6

    _cnt = 0
    while _cnt < dataset_size:
        if pg.draw_question():
            q = pg.generate_question()
            if args.use_cot:
                a = pg.generate_cot_answer()
            else:
                a = pg.generate_answer()
            n_op = pg.Gd.final_num_op
            final_a = pg.Gd.topo[-1].value
            if abs(final_a) >= args.max_solution or \
                (reduce_zero_answer and final_a == 0 and np.random.rand() < reduce_zero_ratio):
                continue
            
            if args.question_aug > 0:
                new_q_list = pg.perturb_question(q, args.question_aug - 1)
                q_list = [q] + new_q_list
                a_list = [a] * args.question_aug
            elif args.answer_aug > 0:
                new_a_list = pg.perturb_answer(args.answer_aug - 1)
                q_list = [q] * args.answer_aug
                a_list = [a] + new_a_list
            else:
                q_list = [q]
                a_list = [a]
            
            for q,a in zip(q_list, a_list):
                questions.append(q)
                if args.content_mask:
                    answers_with_mask.append(a)
                    clean_a, _ = _strip_markers(a)
                    answers.append(clean_a)
                else:
                    answers.append(a)
                num_ops.append(n_op)
                final_answers.append(final_a)
            _cnt += 1
            if _cnt % 5000 == 0:
                print(f"Questions generated: {_cnt}")


    print("Question:")
    print(q)
    print("=" * 50)
    print("Answer:")
    print(a)
    print(f"Final Answer: {final_a}")
    print(f"Number of Operations: {n_op}")
    print("=" * 50)
    print("Ratio of final answer=0:", np.mean(np.array(final_answers) == 0))
    print("total dataset size:", len(questions))

    data_dict = {
        "query": questions,
        "response": answers,
        "num_ops": num_ops,
        "final_answer": final_answers,
    }
    if args.content_mask:
        data_dict["response_with_mask"] = answers_with_mask

    dataset = datasets.Dataset.from_dict(data_dict)

    dataset.save_to_disk(save_path)

    config_file = os.path.join(save_path, 'dataset_config.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(config_file, "w") as f:
        json.dump(config, f, indent=4)