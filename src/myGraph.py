# coding = utf8
from scipy import sparse
import numpy as np
import pandas as pd
from collections import deque


class MyGraph:
    """
    This class implements graph loading from an edge list,
    and provides methods that operate on the graph.

    ------------------------------------------------------------
    Attributes:

    _adjacency_matrix:
        scipy csr matrix (Compressed Sparse Row Matrix)
        use virtual node id as index

    _weighted: bool
        whether the graph is weighted

    _num_vertices: int
        the number of vertices(nodes)

    _num_edges: int
        the number of edges

    _degree: numpy array{float}
        degree of nodes, use virtual node id as index

    _dict_ori_node_id_to_vir_node_id: dict{string: int}
        assign original node id to virtual node id, make sure that id is from 0 to n-1
        this dict return the virtual node id of original node id

    _dict_vir_node_id_to_ori_node_id: dict{int: string}
        assign original node id to virtual node id, make sure that id is from 0 to n-1
        this dict return the original node id of virtual node id
    ------------------------------------------------------------
    Functions:

    read_graph(self, file_name, file_type='edge_list', separator='\t', comment='#')
        read a graph from a file, and initialize the attributes

    approximate_ppr(self, seed_set, alpha=0.15, epsilon=1e-5)
        approximate personalized PageRank vector of the graph

    sweep_with_conductance(self, score, window=3)
        return the best community with the local minimal conductance

    local_community_detection_pagerank_nibble(self, seed_set, alpha=0.15, epsilon=1e-5, window=3)
        Use PageRank Nibble algorithm to find the local community of the seed set.
    ------------------------------------------------------------
    """

    def __init__(self,
                 file_name=None,
                 file_type='edge_list',
                 separator='\t',
                 comment='#'):
        """
        initialize the MyGraph class.

        :param file_name: string
            input file path for building the graph.

        :param file_type: string
            the file type for the input file, currently only support 'edge list'
            Default: 'edge_list'

        :param separator: string
            the separator used in the input file
            Default: '\t'

        :param comment: string
            identifier of the comment
            Default: '#'
        """

        # before initialize, set all attributes to None or Empty
        self._adjacency_matrix = None
        self._weighted = None
        self._num_vertices = None
        self._num_edges = None
        self._degree = None
        self._dict_ori_node_id_to_vir_node_id = dict()
        self._dict_vir_node_id_to_ori_node_id = dict()

        if file_name is None:
            raise Exception('MyGraph.__init__: Edge list file for building graph is None.')

        self.read_graph(file_name, file_type=file_type, separator=separator, comment=comment)

    def read_graph(self,
                   file_name,
                   file_type='edge_list',
                   separator='\t',
                   comment='#'):
        """
        read the graph from input file

        :param file_name: string
            input file path for building the graph.

        :param file_type: string
            the file type for the input file, currently only support 'edge list'
            Default: 'edge_list'

        :param separator: string
            the separator used in the input file
            Default: '\t'

        :param comment: string
            identifier of the comment
            Default: '#'

        :return: None

        """

        if file_type == 'edge_list':
            graph_df = pd.read_csv(file_name, delimiter=separator, comment=comment, header=None)

            if graph_df.shape[1] == 2:
                self._weighted = False
            elif graph_df.shape[1] == 3:
                self._weighted = True
            else:
                raise Exception('MyGraph.read_graph: graph_df.shape[1] not in (2, 3)')

            ori_src = graph_df[0].values
            ori_dst = graph_df[1].values

            vir_src = list()
            vir_dst = list()

            # use virtual node id to replace original node id
            # make sure that virtual node_id is from 0 to n-1
            next_virtual_id = 0
            for ori_node in ori_src:
                if ori_node not in self._dict_ori_node_id_to_vir_node_id:
                    self._dict_ori_node_id_to_vir_node_id[ori_node] = next_virtual_id
                    self._dict_vir_node_id_to_ori_node_id[next_virtual_id] = ori_node
                    next_virtual_id += 1
                vir_src.append(self._dict_ori_node_id_to_vir_node_id[ori_node])
            for ori_node in ori_dst:
                if ori_node not in self._dict_ori_node_id_to_vir_node_id:
                    self._dict_ori_node_id_to_vir_node_id[ori_node] = next_virtual_id
                    self._dict_vir_node_id_to_ori_node_id[next_virtual_id] = ori_node
                    next_virtual_id += 1
                vir_dst.append(self._dict_ori_node_id_to_vir_node_id[ori_node])

            if self._weighted:
                weights = graph_df[2].values
            else:
                weights = np.ones(ori_src.shape[0])

            self._num_vertices = len(self._dict_vir_node_id_to_ori_node_id)
            self._adjacency_matrix = sparse.csr_matrix((weights.astype(np.float64),
                                                        (vir_src, vir_dst)),
                                                       shape=(
                                                           self._num_vertices, self._num_vertices))

            is_symmetric = (self._adjacency_matrix != self._adjacency_matrix.T).sum() == 0
            if not is_symmetric:
                # make the matrix symmetric, new_adj = adj + adj.T
                self._adjacency_matrix = self._adjacency_matrix + self._adjacency_matrix.T
                assert (self._adjacency_matrix != self._adjacency_matrix.T).sum() == 0, \
                    'MyGraph.read_graph: the adjacency matrix is not symmetric.'

            self._num_edges = self._adjacency_matrix.nnz / 2
            self._degree = np.ravel(self._adjacency_matrix.sum(axis=1))
        else:
            raise Exception('MyGraph.read_graph: The type of the input file is not supported '
                            'currently.')

    def approximate_ppr(self,
                        seed_set,
                        alpha=0.15,
                        epsilon=1e-5):
        """
        Compute the approximate Personalized PageRank (PPR) from a set of seed node.

        This function implements method introduced by Andersen et al. in
        "Local Graph Partitioning using PageRank Vectors", FOCS 2006.
        and adjust weighted graph

        :param seed_set: list/set
            list or set of seed node(s)

        :param alpha: float
            Teleportation constant
            alpha corresponds to the probability for the random walk to restarts from the seed set
            Default: 0.15

        :param epsilon: float
            Precision parameter for the approximation
            Default: 1e-5

        :return: numpy 1D array{float}
            Vector containing the approximate PPR for each node of the graph.
        """

        # initialize the ppr vector and residual vector
        ppr = np.zeros(self._num_vertices)
        residual = np.zeros(self._num_vertices)

        # initialize the value of seed(s)
        vir_seed_list = list()
        for seed in seed_set:
            vir_seed_list.append(self._dict_ori_node_id_to_vir_node_id[seed])
        residual[vir_seed_list] = 1. / len(vir_seed_list)

        # initialize push node queue
        push_queue = deque(vir_seed_list)

        # push procedure
        while len(push_queue) > 0:
            cur_node_id = push_queue.pop()
            push_val = residual[cur_node_id] - 0.5 * epsilon * self._degree[cur_node_id]
            residual[cur_node_id] = 0.5 * epsilon * self._degree[cur_node_id]
            ppr[cur_node_id] += alpha * push_val
            put_val = (1. - alpha) * push_val
            for neighbor in self._adjacency_matrix[cur_node_id].indices:
                old_res = residual[neighbor]
                residual[neighbor] += \
                    put_val * self._adjacency_matrix[cur_node_id, neighbor] \
                    / self._degree[cur_node_id]
                threshold = epsilon * self._degree[cur_node_id]
                if residual[neighbor] >= threshold > old_res:
                    push_queue.appendleft(neighbor)

        return ppr

    def sweep_with_conductance(self, score, window=3):
        """
        Return the best community with the local minimal conductance

        This function implements method introduced by Andersen et al. in
        "Local Graph Partitioning using PageRank Vectors", FOCS 2006.

        :param score: numpy array{float}
            The fitness score used to sort nodes in the graph

        :param window: int
            Window parameter used for the detection of a local minimum of conductance,
            means after local minim, there are {window} number greater than it,
            should greater than 1
            Default: 3

        :return: set(int), float
            return the community set, and the corresponding conductance
        """

        total_volume = np.sum(self._degree)
        sorted_nodes = [node for node in range(self._num_vertices) if score[node] > 0]

        if len(sorted_nodes) == 0:
            print "PPR score are all zero."
            return set(), 0.0

        sorted_nodes = sorted(sorted_nodes, key=lambda tmp_node: score[tmp_node], reverse=True)
        sweep_set = set()
        cur_volume = 0.
        cur_cut = 0.
        # best_conductance = 1.
        best_conductance = float('inf')
        best_sweep_set = {sorted_nodes[0]}
        increase_count = 0
        for cur_node in sorted_nodes:
            cur_volume += self._degree[cur_node]
            for neighbor in self._adjacency_matrix[cur_node].indices:
                if neighbor in sweep_set:
                    cur_cut -= self._adjacency_matrix[cur_node, neighbor]
                else:
                    cur_cut += self._adjacency_matrix[cur_node, neighbor]
            sweep_set.add(cur_node)

            if cur_volume == total_volume:
                break
            conductance = cur_cut / min(cur_volume, total_volume - cur_volume)
            if conductance < best_conductance:
                best_conductance = conductance
                # Make a copy of the set
                best_sweep_set = set(sweep_set)
                increase_count = 0
            else:
                increase_count += 1
                if increase_count >= window:
                    break

        # set node id to integer
        best_sweep_set = map(int, best_sweep_set)

        return best_sweep_set, best_conductance

    def local_community_detection_pagerank_nibble(self,
                                                  seed_set,
                                                  alpha=0.15,
                                                  epsilon=1e-5,
                                                  window=3):
        """
        Use PageRank Nibble algorithm to find the local community of the seed set.

        :param seed_set: list/set
            list or set of seed node(s)

        :param alpha: float
            Teleportation constant
            alpha corresponds to the probability for the random walk to restarts from the seed set
            Default: 0.15

        :param epsilon: float
            Precision parameter for the approximation
            Default: 1e-5

        :param window: int
            Window parameter used for the detection of a local minimum of conductance,
            means after local minim, there are {window} number greater than it,
            should greater than 1
            Default: 3

        :return: set(int), float
            return the community set, and the corresponding conductance
        """

        print "---\nBegin Local Community Detection"

        # if seed not in the graph
        for seed in seed_set:
            if seed not in self._dict_ori_node_id_to_vir_node_id:
                print 'Some seed is not contained in the graph'
                empty_result = set()
                return empty_result, 0.0

        # approximate PageRank value
        ppr = self.approximate_ppr(seed_set, alpha=alpha, epsilon=epsilon)
        # sweep to detect the community
        vir_community_set, conductance = self.sweep_with_conductance(ppr, window=window)

        ori_community_set = set()
        for vir_node_id in vir_community_set:
            ori_community_set.add(self._dict_vir_node_id_to_ori_node_id[vir_node_id])

        print "Finished Local Community Detection"
        print "---\n"

        return ori_community_set, conductance