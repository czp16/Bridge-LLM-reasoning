from typing import Dict, List, Tuple, Optional
import random
import os
import json
import numpy as np

from igsm_reasoning.utils import softmax, seed_all, random_select_and_remove, english_dict
from igsm_reasoning.generation.graph_util import Node, StructureGraph, DependencyNode, DependencyGraph

DEFAULT_CONFIG = {
    # "english_path": "igsm_reasoning/generation/english/categorization.json",

    "max_structure_layers": 4,
    "min_items_per_layer": 2,
    "max_items_per_layer": 4,
    "max_instance_in_degree": 4,
    "arithmetic_mod": 10,
    "max_attempts": 50,

    "max_instance_params": 20,
    "max_operations": 20,
    "min_operations": 15,
    "force": False, # force the operation num to be exactly `max_operations`
    "rng_range": 10, # the range of random numbers
    "calculate_mod": False, # whether to calculate the mod value
    "debug": False,

    "reflection_prob": 0.3,
    "analysis_prob_list": [0, 0.1, 0.4, 0.8, 1, 1, 1, 1, 1, 1, 1],
    "shuffle_query": True,
    "redundant_query": True,
    "detailed_computation": True, # whether to show the detailed computation process, e.g., 1+2+3=3+3=6
    "content_mask": False,
}


class ProblemGenerator:
    def __init__(self, config: Dict = {}, seed: Optional[int] = None):
        self.seed(seed)
        self.config = config
        self.debug = config.get("debug", False)
        for k,v in DEFAULT_CONFIG.items():
            if k not in self.config:
                self.config[k] = v

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            seed_all(seed=seed)

    def _load_name_dictionary(self):
        if self.debug:
            self.name_dictionary = [
                [f"W{i}" for i in range(10)],
                [f"X{i}" for i in range(10)],
                [f"Y{i}" for i in range(10)],
                [f"Z{i}" for i in range(10)],
            ]
            self.category_name = ["WWW", "XXX", "YYY", "ZZZ"]
        else:
            # with open(self.config["english_path"], 'r') as f:
            #     all_data = json.load(f)
            all_data = english_dict
            ecosystem_name = random.choice(list(all_data.keys()))
            system = all_data.get(ecosystem_name)
            self.category_name = list(system.keys())
            
            self.name_dictionary = []
            for sub_category in system.values():
                sub_category_name = random.choice(list(sub_category.keys()))
                self.name_dictionary.append(sub_category.get(sub_category_name))

            self.category_name.reverse()
            self.name_dictionary.reverse()
    
    def _get_structuregraph_hp(self, final_num_op: int) -> int:
        """
        Get the hyperparameter for the structure graph.

        Parameters:
        ----------
        final_num_op: int, the final number of operations of the question.
        
        Returns:
        ----------
        num_layers: int, the number of layers for the structure graph.
        w0: int, the minimum number of items per layer
        w1: int, the maximum number of items per layer
        num_edges: int, the number of edges in the structure graph.
        """
        assert self.config["max_structure_layers"] == 4
        max_ip = self.config["max_instance_params"]
        rel = (final_num_op - 1) / (max_ip - 1)
        _weights = np.array([-(rel-0.2)**2, -(rel-0.5)**2, -(rel-0.8)**2])

        num_layers = np.random.choice([2, 3, 4], p=softmax(_weights))
        _t0, _t1 = np.random.choice([2, 3, 4], size=2, replace=True, p=softmax(_weights))
        w0, w1 = min(_t0, _t1), max(_t0, _t1)
        _t0, _t1 = [random.randint((num_layers - 1) * w0, max_ip) for _ in range(2)]
        num_edges = min(_t0, _t1, (num_layers - 1) * w1 * w1)

        return w0, w1, num_layers, num_edges


    
    def draw_question(self) -> bool:
        """
        Randomly generate the structure graph and dependency graph to draw the question.

        Returns:
        ----------
        bool, whether the question is successfully generated
        """
        self._load_name_dictionary()

        max_operations, min_operations = self.config['max_operations'], self.config['min_operations']
        force = self.config['force']
        final_num_op = max_operations if force else min([random.randint(min_operations, max_operations) for _ in range(2)])
        max_op_stage1 = max([random.randint(1, final_num_op) for _ in range(2)])
        max_op_stage2 = random.randint(max_op_stage1, final_num_op)

        w0, w1, num_layers, num_edges = self._get_structuregraph_hp(final_num_op)
        _start_layer = random.randint(0, self.config["max_structure_layers"] - num_layers)
        self.name_dictionary = self.name_dictionary[_start_layer:_start_layer+num_layers]
        self.category_name = self.category_name[_start_layer:_start_layer+num_layers]
        

        flag = False
        _cnt = 0
        while not flag:
            _cnt += 1
            if _cnt > self.config["max_attempts"]: 
                # will retry if stage 3 failed until max_attempts
                return False

            # Generate the structure graph
            Gs = StructureGraph(w0, w1, num_layers, num_edges, self.name_dictionary, self.category_name)
            # Generate the dependency graph
            Gd = DependencyGraph(
                Gs, max_op_stage1, max_op_stage2, final_num_op,
                self.config["max_instance_in_degree"], 
                self.config["rng_range"],
                self.config["arithmetic_mod"],
                self.config["analysis_prob_list"],
            )
            result = Gd.construct_dependency_graph()
            if result == "success":
                flag = True
                self.Gs = Gs
                self.Gd = Gd
            elif result == "stage 4 failed": # will not retry if stage 4 failed
                return False
        return True
        

    def generate_question(self) -> str:
        question_desc = []
        question_node = self.Gd.topo[-1]
        node_list = self.Gd.topo + self.Gd.unnecessary_topo if self.config["redundant_query"] else self.Gd.topo
        if self.debug:
            print("num of redundant nodes:", len(self.Gd.unnecessary_topo))
        for node in node_list:
            if node.node_type == "instance":
                question_desc.append(self.Gd.gen_sentence(node))

        if self.config["shuffle_query"]:
            random.shuffle(question_desc)
        question_desc.append(self.Gd.gen_question(question_node))
        
        final_question = ".\n".join(question_desc)
        return final_question
    
    def perturb_question(self, question: str, n_new_questions: int) -> str:
        """
        Perturb the question by changing the order of the sentences.
        """
        question_desc = question.split(".\n")
        conditions, question = question_desc[:-1], question_desc[-1]
        new_questions = []
        for _ in range(n_new_questions):
            random.shuffle(conditions)
            new_question = conditions + [question]
            new_questions.append(".\n".join(new_question))
        
        return new_questions
    

    def perturb_answer(self, n_new_answers: int) -> str:
        """
        Perturb the answer by generating new topological order.
        """
        old_topo = self.Gd.topo
        question_node = old_topo[-1]

        def _random_topo_sorts(
            old_topo: List[Node] = None,
            k: int = 1,
        ) -> List[List]:
            """
            Generate `k` random topological orders using a
            randomized variant of Kahn’s algorithm.
            Much faster than enumerating all orders when the
            search space is huge.
            """
            results = []
            indeg = {v: 0 for v in old_topo}
            for node in old_topo:
                for p in node.parent_nodes:
                    if p in indeg:
                        indeg[node] += 1
            for _ in range(k):
                _indeg = indeg.copy()
                ready = [v for v, d in indeg.items() if d == 0]
                order = []
                while ready:
                    random.shuffle(ready)
                    v = ready.pop()  # random choice
                    order.append(v)
                    for node in old_topo:
                        if v in node.parent_nodes:
                            _indeg[node] -= 1
                            if _indeg[node] == 0:
                                ready.append(node)
                assert len(order) == len(old_topo), f"len(order): {len(order)}, len(old_topo): {len(old_topo)}"
                # always put the question node at the end
                order.remove(question_node)
                order.append(question_node)
                results.append(order)
            return results
        
        new_topos = _random_topo_sorts(old_topo, n_new_answers)
        new_answers = []
        for new_topo in new_topos:
            self.Gd.topo = new_topo
            # assert len(self.Gd.topo) == len(old_topo), f"len(new_topo): {len(self.Gd.topo)}, len(old_topo): {len(old_topo)}"
            new_answer = self.generate_cot_answer()
            new_answers.append(new_answer)
        self.Gd.topo = old_topo
        return new_answers
    
    def generate_answer(self) -> str:
        answer_desc = []
        answer_desc.append("<think>")
        answer_desc.append("Let's compute the answer step by step.")
        all_variables = [chr(i) for i in range(ord('a'), ord('z') + 1)] # lower case
        all_variables.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)]) # upper case

        for node in self.Gd.topo:
            if not all_variables:
                raise RuntimeError("No enough variable names for answer.")
            _var_name = random_select_and_remove(all_variables)
            answer_desc.append(self.Gd.gen_answer_for_node(node, _var_name, do_mod=self.config["calculate_mod"]))
        answer_desc.append(f"Thus, the answer is {node.value}.")
        answer_desc.append("</think>")
        answer_desc.append(f"<answer>\nThe final answer is \\boxed{{{node.value}}}.\n</answer>")
        
        final_answer = "\n".join(answer_desc)
        return final_answer
    
    def generate_cot_answer(self) -> str:
        use_detailed_computation = self.config["detailed_computation"]
        wrap_content = self.config["content_mask"]

        question_node = self.Gd.topo[-1]
        all_necessary_nodes = self.Gd.topo
        all_unnecessary_nodes = self.Gd.unnecessary_topo
        all_nodes = all_necessary_nodes + all_unnecessary_nodes

        for node in all_nodes:
            node.is_solved = False
            node.is_denoted = False

        all_variables = [chr(i) for i in range(ord('a'), ord('z') + 1)] # lower case
        all_variables.extend([chr(i) for i in range(ord('A'), ord('Z') + 1)]) # upper case
        random.shuffle(all_variables)
        assert len(all_nodes) <= len(all_variables), f"No enough variable names for answer: {len(all_nodes)} nodes > {len(all_variables)} variables."
        node2varname = dict(zip(all_nodes, all_variables[:len(all_nodes)]))

        answer_desc = []
        answer_desc.append("<think>")
        answer_desc.append("Let's compute the answer step by step.")
        
        _curr_idx = 0 # current index of node in Gd.topo
        _last_node_is_correct = False
        while _curr_idx < len(all_necessary_nodes):
            if (0 < _curr_idx < len(all_necessary_nodes) -1) and _last_node_is_correct\
                and random.random() < self.config["reflection_prob"]:
                # if the last node is correct, then there is a chance to reflect
                # it will not select `all_necessary_nodes[_curr_idx]`, instead, it will select a node from
                # either `all_unnecessary_nodes` or the later nodes in `all_necessary_nodes`
                if len(all_unnecessary_nodes) > 0 and random.random() < 0.3:
                    node = random.choice(all_unnecessary_nodes)
                else:
                    _idx = random.randint(_curr_idx + 1, len(all_necessary_nodes) - 1)
                    node = all_necessary_nodes[_idx]
                _var_name = node2varname[node]
                if node.is_solved: # skip if the node has been solved
                    continue
                answer_desc.append(self.Gd.gen_cot_answer_for_node(node, _var_name, use_detailed_computation, wrap_content))
                _last_node_is_correct = node.is_solved
            else:
                # if the last node is incorrect, then it will always select the next node in `all_necessary_nodes`, 
                # i.e., no reflection
                node = all_necessary_nodes[_curr_idx]
                _var_name = node2varname[node]
                if node.is_solved: # has been solved
                    _curr_idx += 1
                    continue
                else:
                    answer_desc.append(self.Gd.gen_cot_answer_for_node(node, _var_name, use_detailed_computation, wrap_content))
                _curr_idx += 1
                _last_node_is_correct = True
        
        answer_desc.append(f"Thus, the answer is {question_node.value}.")
        answer_desc.append("</think>")
        answer_desc.append(f"<answer>\nThe final answer is \\boxed{{{question_node.value}}}.\n</answer>")
        
        final_answer = "\n".join(answer_desc)
        return final_answer


if __name__ == "__main__":
    seed = random.randint(0, 100000)
    # seed = 1000
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    print(f"Seed: {seed}")
    seed_all(seed)
    DEFAULT_CONFIG["debug"] = False
    DEFAULT_CONFIG["min_operations"] = 20
    DEFAULT_CONFIG["max_operations"] = 20
    DEFAULT_CONFIG["reflection_prob"] = 0.0
    DEFAULT_CONFIG["redundant_query"] = True
    DEFAULT_CONFIG["detailed_computation"] = True
    DEFAULT_CONFIG["content_mask"] = False

    pg = ProblemGenerator(DEFAULT_CONFIG)
    flag = True
    while flag:
        if pg.draw_question():
            q = pg.generate_question()
            a_cot = pg.generate_cot_answer()
            # a = pg.generate_answer()
            n_op = pg.Gd.final_num_op
            final_a = pg.Gd.topo[-1].value
            # if abs(final_a) >= 1000:
            #     continue
            flag = False
            print(q)
            print("=" * 50)
            print(a_cot)
            # print("=" * 50)
            # print(a)
            print(f"Final Answer: {final_a}")
            print(f"Number of Operations: {n_op}")

            # q_new = pg.perturb_question(q)
            # print("=" * 50)
            # print(q_new)
            a_new = pg.perturb_answer(3)
            print("=" * 50)
            print(a_new[0])
            print("=" * 50)
            print(a_new[-1])
    # else:
    #     print("Failed to generate the question.")

    # _cnt = 0

    # pg = ProblemGenerator(DEFAULT_CONFIG)
    # for _ in range(100):
    #     if pg.draw_question():
    #         _cnt += 1

    # print(f"Success rate: {_cnt} / {100}")