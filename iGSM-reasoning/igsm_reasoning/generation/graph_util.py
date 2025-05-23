"""
define the node and graph classes
"""
from typing import List, Optional, Tuple, Dict, Any, Literal, Union
from collections import deque
import random
import numpy as np
from igsm_reasoning.utils.misc import softmax, random_select_and_remove, seed_all, simplify_equation


class Node:
    """
    Node representing a node in a `StructureGraph`. Each node has an index, which is a tuple of
    (layer index, node index), e.g., (li, ni) means the ni-th node in the li-th layer.
    """
    def __init__(
        self,
        parent_nodes: List["Node"], 
        index: Tuple[int, int],
        name: Optional[str] = None,
    ):
        """
        Parameters:
        ------------
        parent_nodes: the parent nodes that it depends on, e.g., if there are school backpacks in the \
            art classroom, then school backpack is one of the parent nodes of the art classroom.
        index: the index of the node, which is a tuple of (layer index, node index).
        name: the name of the node, i.e., "school backpack" or "art classroom".
        """
        self.parent_nodes = parent_nodes
        self.index = index
        self.name = name

    @property
    def in_degree(self):
        return len(self.parent_nodes)

class Parameter(Node):
    """
    Parameter representing a parameter derived from a `StructureGraph`. There are two types of parameters:
    - instance parameter: defined by each edge, e.g., the edge `((li, ni), (lj, nj))` defines a parameter
        which means "the number of `nodes[li][ni].name` in each `nodes[lj][nj].name`", represented as 
        `((li, ni), (lj, nj))`.
    - abstract parameter: not explicitly defined by the graph, but it is the parameter describing
        the total number of items in one category w.r.t. a node in a higher layer. E.g., one abstract
        parameter is "the total number of [Category of layer li] in each `nodes[lj][nj].name`",
        represented as `(li, (lj, nj))`.
    
    See the `StructureGraph` class for more details.
    """
    def __init__(
        self, 
        parent_nodes: List["Parameter"], 
        index: Tuple[int | Tuple[int, int], Tuple[int, int]],
        name: str,
    ):
        """
        Parameters:
        ------------
        parent_nodes: the parent parameters that it depends on.
        index: the index of the parameter, which is a tuple. For instance parameter, it is `((li, ni), (lj, nj))`,\
            and for abstract parameter, it is `(li, (lj, nj))`.
        name: the name of the parameter.
        """
        self.parent_nodes = parent_nodes
        self.index = index
        self.name = name
    
    @property
    def param_type(self) -> Literal["instance", "abstract"]:
        return "abstract" if isinstance(self.index[0], int) else "instance"
    
    @property
    def d_level(self) -> int:
        """
        Difficulty level of abstract parameter, which is the difference of the layer index
        """
        return self.index[1][0] - self.index[0] if self.param_type == "abstract" else -1

class DependencyNode(Node):
    """
    DependencyNode representing a node in a `DependencyGraph`. Each node corresponds to a parameter from
    the `StructureGraph`. 
    """
    def __init__(
        self,
        parent_nodes: List["DependencyNode"],
        node_type: Literal["instance", "abstract", "rng"],
        eval_equation: str = "", 
        name: Optional[str] = None,
        value: int = 0,
    ):
        """
        Parameters:
        ------------
        parent_nodes: the parent nodes that it depends on.
        node_type: the type of the node, either "instance", "abstract" or "rng" (random number generator), corresponding \
            to the parameter type.
        eval_equation: the equation that the node performs from parent nodes, e.g., f"{parent[0].value}+{parent[1].value}".
        name: the name of the node.
        value: the correct value of the node by performing the equation from parent nodes.
        """
        self.parent_nodes = parent_nodes
        self.node_type = node_type
        self.eval_equation = eval_equation
        self.name = name
        self.value = value
        self.var_name = "" # only used for answer generation
        self.question_desc: str = ""
        self.is_solved = False # whether the node is solved, used for answer generation
        self.is_denoted = False # whether the node is denoted, e.g., "let's denote the number of ... as x"

    def get_equation(self):
        """
        Get the operation that the node performs from parent nodes, only for abstract nodes.
        """
        assert self.node_type == "abstract", f"{self.node_type} node does not use this function"

        if len(self.parent_nodes) > 1 and self.parent_nodes[1].node_type == "abstract": 
            # difficult level >= 2, should be {0} * {1} + {2} * {3} + ...
            # then we will get the value by `eval(eval_equation.format(*[p.value for p in self.parent_nodes]))`
            self.eval_equation = " + ".join([f"{{{i}}} * {{{i+1}}}" for i in range(0, len(self.parent_nodes), 2)])
        else:
            # difficult level = 1, should be {0} + {1} + {2} + ...
            self.eval_equation = " + ".join([f"{{{i}}}" for i in range(len(self.parent_nodes))])


class StructureGraph:
    """
    StuctureGraph representing the structure (a DAG) of the variables of a system (e.g., the fig 1 left
    in the paper), where the problem is built on:
    - The nodes in the graph are the variables, e.g., school backpack, art classroom, etc.
    - The edges are the dependencies, indicating the relationship between the variables.

    Based on the structure graph, we can get parameters (e.g., each art classroom's school backpack) and
    construct the dependency graph, which is the graph leading to the final problem.
    """
    def __init__(
        self, 
        w0: int, 
        w1: int, 
        num_layers: int, 
        num_edges: int,
        name_dictionary: List[List[str]],
        layer_category_name: List[str],
    ):
        """
        Parameters:
        ------------
        w0: int, the minimum number of items per layer
        w1: int, the maximum number of items per layer
        num_layers: int, the number of layers
        num_edges: int, the number of edges
        names: list of names for each layer
        layer_category_name: list of category name for each layer
        """
        
        assert num_layers == len(name_dictionary) == len(layer_category_name), \
            f"Number of layers and names do not match: {num_layers}, {len(name_dictionary)}, {len(layer_category_name)}"
        self.w0 = w0
        self.w1 = w1
        self.num_layers = num_layers
        self.num_edges = num_edges

        self.layer_category_name = layer_category_name

        self.draw_structure()
        self.name_nodes(name_dictionary)

        self.construct_param_dependency_graph()
        

    def draw_structure(self):
        """
        Algorithm 1 in the paper. Randomly draw the structure of the graph.
        """
        w0, w1, num_layers, num_edges = self.w0, self.w1, self.num_layers, self.num_edges

        # 1. Randomly generate the number of items per layer
        
        n_items_per_layer = [w0 for _ in range(num_layers)]
        _prob = random.random()
        while True:
            if all([n_items_per_layer[i] == w1 for i in range(num_layers)]):
                break
            e_minus = sum(n_items_per_layer[1:])
            e_plus = sum(n_items_per_layer[i]*n_items_per_layer[i+1] for i in range(num_layers-1))
            if e_minus == num_edges:
                break
            elif e_plus < num_edges or random.random() < _prob:
                idx = random.choice([i for i in range(num_layers) if n_items_per_layer[i] < w1])
                n_items_per_layer[idx] += 1
            else:
                break

        # 2. Randomly generate the nodes and edges
        # list of nodes in each layer, looking like [[x0,x1,x2], [y0,y1,y2,y3], ...]
        # then we can get the node by self.nodes[li][ni]
        self.nodes: List[List[Node]] = [] 
        for i, n_item in enumerate(n_items_per_layer):
            self.nodes.append([Node(index=(i,j), parent_nodes=[]) for j in range(n_item)])
        
        # 2.1 generate minimum number (e_minus) of edges
        # each edge is represented as ((li, ni), (lj, nj))
        # where li, lj are the layer index and ni, nj are the node index
        self.edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for i in range(1, num_layers):
            for j in range(n_items_per_layer[i]):
                k = random.randint(0, n_items_per_layer[i-1]-1)
                self.edges.append(((i-1, k), (i, j)))
        
        # 2.2 generate the remaining edges
        n_current_edges = len(self.edges)
        while n_current_edges < num_edges:
            i = random.randint(1, num_layers-1)
            j = random.randint(0, n_items_per_layer[i]-1)
            k = random.randint(0, n_items_per_layer[i-1]-1)
            if ((i-1, k), (i, j)) not in self.edges:
                self.edges.append(((i-1, k), (i, j)))
                n_current_edges += 1

        # 2.3 add parent nodes to each node according to the edges
        for (li, ni), (lj, nj) in self.edges:
            self.nodes[lj][nj].parent_nodes.append(self.nodes[li][ni])
    

    def name_nodes(self, name_dictionary: List[List[str]]):
        """
        Randomly assign names to each node in the graph.
        """
        for i, layer in enumerate(self.nodes):
            names_per_layer = random.sample(name_dictionary[i], len(layer))
            for node, name in zip(layer, names_per_layer):
                node.name = name
    

    def construct_param_dependency_graph(self):
        """
        Construct the dependency graph of all instance/abstract parameters from the structure graph.
        """
        self.params: List[Parameter] = []
        self.params.extend(self.get_all_instance_param())
        self.params.extend(self.get_all_abstract_param())
        
        index2param = {param.index: param for param in self.params}

        # add the dependency parameters to each abstract parameter
        for param in self.params:
            if param.param_type == "abstract":
                li, (lj, nj) = param.index
                if lj - li == 1:
                    # the parent parameters of the abstract parameter are the instance parameters defined by the 
                    # edges between the layers li and lj
                    for parent in self.nodes[lj][nj].parent_nodes:
                        # in this case, the value of the abstract parameter is the sum of its parents
                        # i.e., the operation (will be defined in the DependencyGraph) is f"{parent.value} + ..."
                        # TODO: remove the assert statement if the graph is correct
                        # assert (parent.index, (lj, nj)) in self.edges, f"Invalid edge: {parent.index} -> {(lj,nj)}"
                        param.parent_nodes.append(index2param[(parent.index, (lj, nj))])
                        
                elif lj - li > 1:
                    # the parent parameters of the abstract parameter include both instance and abstract parameters
                    for parent in self.nodes[lj][nj].parent_nodes:
                        # in this case, the value of the abstract parameter is the sum of its instance parent times
                        # the abstract parent, i.e., the operation is f"{parents[0].value} * {parents[1].value} + ..."
                        # TODO: remove the assert statement if the graph is correct
                        # assert (parent.index, (lj, nj)) in self.edges, f"Invalid edge: {parent.index} -> {(lj,nj)}"
                        param.parent_nodes.append(index2param[(parent.index, (lj, nj))]) # instance parameter
                        param.parent_nodes.append(index2param[(li, parent.index)]) # abstract parameter
                else:
                    raise ValueError(f"Invalid parameter index: {param.index}")

        
    
    def get_all_instance_param(self):
        """
        Instance parameters are defined by each edge, e.g., the edge `((li, ni), (lj, nj))` defines
        the instance parameter "the number of `nodes[li][ni].name` in each `nodes[lj][nj].name`".
        """
        all_instance_param = []
        for edge in self.edges:
            (li, ni), (lj, nj) = edge
            all_instance_param.append(Parameter([], edge, name=f"each {self.nodes[lj][nj].name}'s {self.nodes[li][ni].name}"))
        return all_instance_param
    

    def get_all_abstract_param(self):
        """
        Abstract parameters are not directly defined by the graph, they are the parameters
        describing the number of all items in the category in a node which is in a higher layer.
        E.g., suppose the category name of layer li is backpack and each node in layer li denotes
        a specific backpack (e.g., `nodes[li][ni].name` is school backpack), and the `nodes[lj][nj].name`
        is "art classroom". \
        Then there is an abstract parameter is 'the total number of backpack in each art classroom'.
        """
        all_abstract_param = []
        for lj, layer in enumerate(self.nodes):
            if lj == 0:
                continue
            for li in range(lj): # li: index of the lower layer
                for nj, _ in enumerate(layer):
                    param = Parameter([], (li, (lj, nj)), name=f"each {self.nodes[lj][nj].name}'s {self.layer_category_name[li]}")
                    all_abstract_param.append(param)
        return all_abstract_param
    

class DependencyGraph:
    """
    DependencyGraph representing the dependency graph of the parameters from the `StructureGraph`, corresponding 
    to fig 1 right in the paper. We will then generate the problem based on the dependency graph.
    - The nodes in the graph are the parameters, e.g., "each art classroom's school backpack".
    - The edges are the dependencies, e.g., "each art classroom's school backpack" depends on "each film studio's \
        messenger backpack" by "each art classroom's school backpack" = "each film studio's messenger backpack" * 2.
    """
    def __init__(
        self, 
        Gs: StructureGraph, 
        max_op_stage1: int, 
        max_op_stage2: int,
        final_num_op: int,
        max_instance_in_degree: int = 4,
        rng_range: Union[Tuple[int, int], int] = 10,
        mod: int = 23,
        analysis_prob_list: Optional[List[float]] = None, 
    ):
        """
        Parameters:
        ------------
        Gs: the structure graph
        max_op_stage1: the maximum number of operations in the first stage
        max_op_stage2: the maximum number of operations in the second stage
        final_num_op: the final number of operations desired
        max_instance_in_degree: the maximum in-degree of instance nodes
        rng_range: the range of the random number generator
        mod: the mod number during the computation
        analysis_prob_list: The probability of repeating the condition of this node in the answer, i.e., whether to add `analysis_str` to the answer
        """
        self.Gs = Gs

        self.max_op_stage1 = max_op_stage1
        self.max_op_stage2 = max_op_stage2
        self.final_num_op = final_num_op
        self.max_instance_in_degree = max_instance_in_degree
        self.rng_range = rng_range if isinstance(rng_range, tuple) else (0, rng_range)
        self.mod = mod
        self.analysis_prob_list = analysis_prob_list
        
        self.num_op = 0 # the current number of operations

        self.rng = DependencyNode([], "rng", name="RNG", value=0)
        self.topo: List[DependencyNode] = []

    def construct_dependency_graph(self) -> Literal["success", "stage 3 failed", "stage 4 failed"]:
        """
        Construct the dependency graph of the structure graph.

        Return:
        -------
        str: whether the graph is successfully constructed
        """
        # This dictionary will map from a Parameter to its corresponding DependencyNode.
        self.param2depnode: Dict[Parameter, DependencyNode] = {}
        # We will get all DependencyNodes in first 2 stages.
        self.construct_Gd1()
        self.construct_Gd2()
        self.depnodes = list(self.param2depnode.values()) # correct

        if not self.construct_Gd3():
            # print("Failed to construct Gd3")
            return "stage 3 failed"
        if not self.construct_Gd4():
            # print("Failed to construct Gd4")
            return "stage 4 failed"
        self.construct_Gd()
        # self.depnodes = list(self.param2depnode.values()) # update unnecessary nodes
        return "success"



    def _add_ascendant_nodes_to_graph(self, root_param: Parameter):
        """
        Add all ascendant nodes of the root_param (including itself) to the dependency graph.
        Meanwhile keep the structure (parent-child relationship) of the nodes.

        Return:
        -------
        bool: whether the max_op_stage1 is reached
        """
        # temperary dictionary to map from Parameter to DependencyNode
        # only store the newly added DependencyNodes, no overlap with self.param2depnode
        tmp_param2depnode: Dict[Parameter, DependencyNode] = {}

        # First pass: 
        # Use BFS to create all DependencyNodes without linking their parents.
        open_list = deque([root_param])
        while open_list:
            param = open_list.popleft()

            # If we haven't created a node for this parameter yet, create it now.
            if param not in self.param2depnode and param not in tmp_param2depnode:
                dep_node = DependencyNode(
                    parent_nodes=[],  # will be linked later
                    node_type=param.param_type,
                    name=param.name,
                )
                
                tmp_param2depnode[param] = dep_node

                for parent_param in param.parent_nodes:
                    if parent_param not in self.param2depnode and parent_param not in tmp_param2depnode:
                        open_list.append(parent_param)

        # check whether the max_op_stage1 is reached
        new_op = sum([max(1, param.in_degree - 1) for param in tmp_param2depnode.keys()])
        if self.num_op + new_op > self.max_op_stage1:
            return False
        
        # if not reached, add the new DependencyNodes to the graph
        self.num_op += new_op
        self.param2depnode.update(tmp_param2depnode)
        # Second pass: 
        # Now that the new DependencyNodes are created, link them together.
        for param, dep_node in tmp_param2depnode.items():
            dep_node.parent_nodes = [self.param2depnode[parent_param] for parent_param in param.parent_nodes]
            if dep_node.node_type == "abstract":
                dep_node.get_equation()
        
        return True
    
    def construct_Gd1(self):
        """
        Stage 1 of the dependency graph construction:

        Randomly select abstract parameters and its all dependent instance/abstract parameters
        recursively until the number of operations reaches `max_op_stage1`.
        """
        flag = True
        while flag:
            flag = False
            # select difficulty level from high to low to make the op_num as close to max_op_stage1 as possible
            for lvl in reversed(range(1, self.Gs.num_layers)):
                # TODO: maybe can optimize the valid_params selection, don't need to iterate all params every time
                valid_params = [p for p in self.Gs.params if p.d_level == lvl and p not in self.param2depnode]
                if valid_params:
                    selected_param = random.choice(valid_params)
                    if self._add_ascendant_nodes_to_graph(selected_param):
                        flag = True
                        break
                    

    def construct_Gd2(self):
        """
        Stage 2 of the dependency graph construction:

        Randomly select instance parameters until the number of operations reaches `max_op_stage2`.
        """
        remaining_instance_params = [p for p in self.Gs.params if p.param_type == "instance" and p not in self.param2depnode]
        while self.num_op < self.max_op_stage2 and remaining_instance_params:
            # randomly remove a parameter from the remaining_instance_params
            selected_param: Parameter = random_select_and_remove(remaining_instance_params)
            self.param2depnode[selected_param] = DependencyNode(
                # though they should have no parents at this stage
                parent_nodes=[self.param2depnode[p] for p in selected_param.parent_nodes],
                node_type="instance",
                name=selected_param.name,
            )
            self.num_op += max(1, selected_param.in_degree - 1)


    def construct_Gd3(self) -> bool:
        """
        Stage 3 of the dependency graph construction:

        Get a random topological order of the nodes in the graph.
        
        Return:
        -------
        bool: whether the graph is successfully constructed
        """
        # here we only compute the out degree to the remaining nodes
        # i.e., exclude the degree to topo
        out_degree_map = {node: 0 for node in self.depnodes}
        for node in self.depnodes:
            for parent in node.parent_nodes:
                out_degree_map[parent] += 1
        
        # start from the leave nodes
        # topo: the list of DependencyNodes in topological order 
        # (first it is from leaves to roots, will be reversed later)
        # group1: the parent nodes of nodes in topo, corresponding to Next1_Gd in paper
        # group2: the nodes having no children in remaining_nodes (out_degree = 0) corresponding to Next2_Gd in paper
        # IMPORTANT: we don't use data structure `set` here because the order cannot be guaranteed to be the same 
        # given a fixed seed
        remaining_nodes = self.depnodes[:]
        topo = self.topo
        group1 = []
        group2 = [node for node in remaining_nodes if out_degree_map[node] == 0]
        
        while True:
            if not topo:
                node = random.choice(group2)
            else:
                node = random.choice([node for node in group1 if node in group2]) # the intersection of group1 and group2
            topo.append(node)
            if node in remaining_nodes:
                remaining_nodes.remove(node)
            if node in group1:
                group1.remove(node)
            if node in group2:
                group2.remove(node)
            for p in node.parent_nodes:
                group1.append(p)
                out_degree_map[p] -= 1
                if out_degree_map[p] == 0:
                    group2.append(p)

            if not remaining_nodes:
                break
            if not any(_node in group2 for _node in group1): # no intersection
                if node.node_type == "abstract":
                    return False
                # non-uniformly random select a parent node from group2
                # weight of non-uniform random selection
                group2_list = list(group2)
                _g = abs(np.random.randn())
                _w = [int(node.node_type == "abstract") + int(node in group1) for node in group2_list]
                _w = softmax(np.array(_w) * _g)
                p_node = np.random.choice(group2_list, p=_w) # non-uniform random selection

                # and create an edge p_node -> node; note `node` is not in remaining_nodes
                # so p_node is still in group2; meanwhile p_node will be added to group1
                # then p_node \in group1 \cap group2
                node.parent_nodes.append(p_node)
                group1.append(p_node)
                
            elif node.node_type == "instance":
                if random.random() < 0.5:
                    # non-uniformly random select a parent node, same as above
                    _g = abs(np.random.randn())
                    _w = [int(node.node_type == "abstract") + int(node in group1) for node in remaining_nodes]
                    _w = softmax(np.array(_w) * _g)
                    p_node = np.random.choice(remaining_nodes, p=_w) # non-uniform random selection

                    # and create an edge p_node -> node
                    node.parent_nodes.append(p_node)
                    group1.append(p_node)

        topo.reverse()
        return True


    def construct_Gd4(self) -> bool:
        """
        Stage 4 of the dependency graph construction:

        Add additional dependency edges to make the graph more complex.
        Specifically, the in-degree of instance nodes should between 1 and 
        `self.max_instance_in_degree` and the final number of operations are
        exactly `self.final_num_op`.

        Return:
        -------
        bool: whether the graph is successfully constructed
        """
        topo = self.topo
        curr_num_op = [max(1, node.in_degree - 1) for node in topo]
        max_num_op = [min(self.max_instance_in_degree - 1, max(1,i)) for i in range(len(topo))]
        while sum(curr_num_op) < self.final_num_op:
            # randomly select a node to increase its in-degree
            candidates = [i for i, node in enumerate(topo) if curr_num_op[i] < max_num_op[i] and node.node_type == "instance"]
            if not candidates:
                return False
            curr_num_op[random.choice(candidates)] += 1
        

        for i, node in enumerate(topo):
            if node.node_type == "instance":
                pool = topo[:i] + [self.rng]
                
                dep_num = random.randint(1,2) if curr_num_op[i] == 1 else curr_num_op[i] + 1
                dep_num = min(dep_num, len(pool))
                if node.parent_nodes:
                    assert len(node.parent_nodes) == 1
                    pool.remove(node.parent_nodes[0])
                    dep_num -= 1
                if dep_num == len(pool):
                    node.parent_nodes.extend(pool)
                else:
                    if random.random() < 0.5:
                        node.parent_nodes.append(self.rng)
                        pool.remove(self.rng)
                        dep_num = max(0, dep_num - 1) # otherwise dep_num will be -1
                    node.parent_nodes.extend(random.sample(pool, dep_num))

        return True
    
    def construct_Gd(self):
        """
        Stage 5 of the dependency graph construction:
        Add unnecessary parameters to make the graph more complex. Specifically, add all 
        remaining instance parameters to the graph. Meanwhile, we may add unnecessary
        abstract parameters.
        """
        self.unnecessary_topo = []

        remaining_instance_params = [p for p in self.Gs.params if p.param_type == "instance" and p not in self.param2depnode]
        remaining_abstract_params = [p for p in self.Gs.params if p.param_type == "abstract" and p not in self.param2depnode]

        unnecessary_nodes = [] # unnecessary instance parameters
        while remaining_instance_params:
            # computable_params: corresponding to K in paper
            computable_params: List[DependencyNode | Parameter] = self.depnodes[:] 
            # select computable abstract parameter from the `remaining_abstract_params`
            # they are Parameter instead of DependencyNode; will create DependencyNodes for them when added to the graph
            for p in remaining_abstract_params:
                if all([p_parent in self.param2depnode for p_parent in p.parent_nodes]):
                    computable_params.append(p)
            
            param_a = random_select_and_remove(remaining_instance_params)
            assert not param_a.parent_nodes, f"Invalid parameter: {param_a.name} has parent nodes"
            node_a = DependencyNode(
                # though it should have no parents at this stage
                parent_nodes=[],
                node_type="instance",
                name=param_a.name,
            )
            self.param2depnode[param_a] = node_a
            
            # randomly select the dependencies of the instance parameter:
            # they can be either from the computable_nodes (including unnecessary abstract parameters) 
            # or the newly added unnecessary instance parameters
            if random.random() < 0.5: # select from newly added unnecessary instance parameters
                pool = unnecessary_nodes + [self.rng]
                unnecessary_nodes.append(node_a)
            else: # including unnecessary abstract parameters
                pool = computable_params + [self.rng]

            dep_num = 1
            while dep_num < min(self.max_instance_in_degree, len(pool)) and random.random() < 0.5:
                dep_num += 1
            if dep_num == len(pool):
                node_a_parents = pool
            else: # 50% chance to choose the RNG node and 50% to exclude it
                node_a_parents = []
                add_rng = False
                if random.random() < 0.5:
                    add_rng = True
                    dep_num -= 1
                pool.pop() # pop the RNG node
                node_a_parents.extend(random.sample(pool, dep_num))
                if add_rng:
                    node_a_parents.append(self.rng)

            for p in node_a_parents:
                if isinstance(p, DependencyNode):
                    node_a.parent_nodes.append(p)
                elif isinstance(p, Parameter):
                    # create a DependencyNode for the unnecessary abstract parameter
                    node = DependencyNode(
                        parent_nodes=[self.param2depnode[p_parent] for p_parent in p.parent_nodes],
                        node_type="abstract",
                        name=p.name,
                    )
                    node.get_equation()
                    # and add it to graph
                    self.param2depnode[p] = node
                    node_a.parent_nodes.append(node)

                    remaining_abstract_params.remove(p)
                    self.unnecessary_topo.append(node)
            
            self.unnecessary_topo.append(node_a)
    
    def gen_sentence(self, node: DependencyNode) -> str:
        """
        Generate the Discription for the DependencyNode.
        """
        assert node.node_type == "instance", f"{node.node_type} node does not have a description"

        
        desc = f"The number of {node.name} equals"
        _equation = ""
        # the order of the parents may be shuffled
        parents = node.parent_nodes
        has_rng = self.rng in parents

        _suffix = ""

        if has_rng:
            random_int = random.randint(self.rng_range[0], self.rng_range[1] - 1)
            desc += " " + str(random_int)
            _equation += str(random_int)
            parents.remove(self.rng) # will add it back later
            if parents:
                _op = random.choice([" + ", " * "])
                _op_str = " more than" if _op == " + " else " times"
                desc += _op_str
                _equation += _op
                if len(parents) > 1:
                    _equation += "("
                    _suffix = ")"
                    
        if len(parents) == 1:
            desc += f" {parents[0].name}"
            _equation += "{0}"
        elif len(parents) == 2:
            random.shuffle(parents)
            _op = random.choice(["+", "-"])
            _op_str = "sum" if _op == "+" else "difference"
            desc += f" the {_op_str} of {parents[0].name} and {parents[1].name}"
            _equation += f"{{{0}}} {_op} {{{1}}}" # should be "{0} + {1}" or "{0} - {1}"
        elif len(parents) > 2:
            random.shuffle(parents)
            desc += " the sum of " + ", ".join([p.name for p in parents[:-1]]) + f" and {parents[-1].name}"
            _equation += " + ".join([f"{{{i}}}" for i in range(len(parents))])

        _equation += _suffix
        node.eval_equation = _equation
        node.question_desc = desc
        if has_rng:
            parents.append(self.rng)
        return desc

    def gen_question(self, node: DependencyNode) -> str:
        """
        Generate the question for the DependencyNode.
        """
        assert node.node_type in ["instance", "abstract"], f"{node.node_type} node does not have a question"
        args = node.name.split("'s ")
        arg0, arg1 = args[0], args[1]
        desc = f"How many {arg1} does {arg0} have?"
        return desc

    def gen_answer_for_node(self, node: DependencyNode, variable_name: str, do_mod: bool) -> str:
        """
        Generate the answer for the DependencyNode.

        Parameters:
        ------------
        node: the DependencyNode
        variable_name: the name of the variable in the answer for simplicity, it can be "a-z" or "A-Z"
        """
        assert node.node_type in ["instance", "abstract"], f"{node.node_type} node does not have an answer"
        node.var_name = variable_name
        parents = node.parent_nodes if node.parent_nodes[-1] != self.rng else node.parent_nodes[:-1]
        
        # only plug in the var_name of parent nodes, e.g., b + 3
        eval_equation1 = node.eval_equation.format(*[p.var_name for p in parents]) 
        # only plug in the val of parent nodes, e.g., 5 + 3 (suppose b = 5)
        eval_equation2 = node.eval_equation.format(*[p.value for p in parents]) 
        # final val of the node
        node.value = eval(eval_equation2) 
        if do_mod:
            node.value = node.value % self.mod

        # print(f"node {node.name}: var_name: {node.var_name}, eval_eq: {node.eval_equation}, parents: {[p.name for p in node.parent_nodes]}, value: {node.value}")

        if parents:
            if any([_op in eval_equation2 for _op in ["+", "-", "*"]]):
                solution = f"{variable_name} = {eval_equation1} = {eval_equation2} = {node.value}"
            else:
                solution = f"{variable_name} = {eval_equation1} = {node.value}"
        else:
            solution = f"{variable_name} = {node.value}"

        answer_desc = f"Define {node.name} as {variable_name}. So {solution}."
        return answer_desc
    
    def gen_cot_answer_for_node(
        self, 
        node: DependencyNode, 
        variable_name: str, 
        detailed_computation: bool = True,
        wrap_content: bool = False,
    ) -> str:
        """
        Generate the answer for the DependencyNode.

        Parameters:
        ------------
        node: the DependencyNode
        variable_name: the name of the variable in the answer for simplicity, it can be "a-z" or "A-Z"
        detailed_computation: whether to show the detailed computation
        wrap_content: whether to wrap the content in the answer by << and >>. The content includes the node name, the equation, and the final value.
        """
        assert node.node_type in ["instance", "abstract"], f"{node.node_type} node does not have an answer"
        assert not node.is_solved, "The node has been solved"
        node.var_name = variable_name
        # `parents` is the parent nodes except the RNG node
        parents = node.parent_nodes if node.parent_nodes[-1] != self.rng else node.parent_nodes[:-1]

        # The probability of repeating the condition of this node in the answer, i.e., whether to add `analysis_str`
        # if the node has i parents, the probability is `analysis_prob_list[i]`
        analysis_prob = self.analysis_prob_list[len(node.parent_nodes)]
        is_correct = all([p.is_solved for p in parents]) # it should be true if parents is empty

        if is_correct:
            # only plug in the var_name of parent nodes, e.g., b + 3
            eval_equation1 = node.eval_equation.format(*[p.var_name for p in parents]) 
            # only plug in the val of parent nodes, e.g., 5 + 3 (suppose b = 5)
            eval_equation2 = node.eval_equation.format(*[p.value for p in parents])
            node.value = eval(eval_equation2)
            node.is_solved = True
        
        if wrap_content and is_correct:
            node_name = f"<<{node.name}>>"
        else:
            node_name = node.name

        direct_extract_str = [
            f"According to the information given, the number of {node_name} is {node.value}.",
            f"The number of {node_name} is {node.value}.",
        ]

        node_select_str = [
            f"We can then calculate the number of {node_name}.",
            f"Now, we can find the number of {node_name}.",
        ]

        define_str1 = [
            f"Let it be {variable_name}.",
            f"Let's denote it as {variable_name}.",
            f"Let's call it {variable_name}.",
        ]

        define_str2 = [
            f"Next, let {variable_name} represent the number of {node_name}.",
            f"Then, let's denote the number of {node_name} as {variable_name}.",
            f"Next, let's define the number of {node_name} as {variable_name}.",
        ]

        denoted_define_str = f"Remember that it has been denoted as {variable_name}."

        if node.node_type == "instance":
            analysis_str = f"We know that it equals {node.question_desc.split('equals')[-1].strip()}."
        else:
            analysis_str = "We know that it = " + node.eval_equation.format(*[p.name for p in parents]) + "."
        
        if not is_correct:
            unknown_parents = [p for p in parents if not p.is_solved]
            be_word = "is" if len(unknown_parents) == 1 else "are"
            incorrect_analysis_str = [
                f"Wait, the number of {', '.join([p.name for p in unknown_parents])} {be_word} unknown, so we cannot compute it.",
                f"However, we don't know the number of {', '.join([p.name for p in unknown_parents])}, thus we cannot calculate it now.",
                f"But we haven't calculated the number of {', '.join([p.name for p in unknown_parents])} yet, thus the value of {variable_name} is still unknown.",
            ]

        if parents and is_correct:
            if detailed_computation:
                compute_steps = simplify_equation(eval_equation2)
                assert node.value == int(compute_steps[-1].strip())
            else: # no detailed computation
                compute_steps = [eval_equation2]
                if eval_equation2.strip() != str(node.value):
                    compute_steps.append(str(node.value))

            if not wrap_content:
                _compute_steps = " = ".join(compute_steps)
                equation_str = f"Then {variable_name} = {eval_equation1} = {_compute_steps}."
            else:
                if len(compute_steps) >= 3: # there are intermediate steps, we will not wrap them as content
                    intermediate_steps = " = ".join(compute_steps[1:-1])
                    equation_str = f"Then {variable_name} = <<{eval_equation1} = {compute_steps[0]} =>> {intermediate_steps} = <<{node.value}>>."
                else:
                    _compute_steps = " = ".join(compute_steps)
                    equation_str = f"Then {variable_name} = <<{eval_equation1} = {_compute_steps}>>."
            
        else:
            if wrap_content:
                equation_str = f"So {variable_name} = <<{node.value}>>."
            else:
                equation_str = f"So {variable_name} = {node.value}."

        if not parents:
            answer_desc = f"- {random.choice(direct_extract_str)} {random.choice(define_str1)} {equation_str}"
        else:
            # select this node and denote it
            if not node.is_denoted: # it's the first time to denote the variable
                if random.random() < 0.5: # use prefix like "Now, we can calculate..."
                    answer_desc = f"- {random.choice(node_select_str)} {random.choice(define_str1)}"
                else:
                    answer_desc = f"- {random.choice(define_str2)}"
            else:
                answer_desc = f"- {random.choice(node_select_str)} {denoted_define_str}"
            node.is_denoted = True

            # add the analysis of the node
            if node.node_type == "instance" and random.random() < analysis_prob:
                answer_desc += f" {analysis_str}"
            
            # if it's solvable, add the equation; else add the analysis about why it's not solvable
            if not is_correct:
                answer_desc += f" {random.choice(incorrect_analysis_str)}"
            else:
                answer_desc += f" {equation_str}"
        return answer_desc