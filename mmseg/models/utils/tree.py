

import queue

import numpy as np
import math
import networkx as nx
import torch
import matplotlib.pyplot as plt

from .visualize_helpers import colour_nodes, plot_hierarchy
from .hierarchy_helpers import json2rels,hierarchy_pos

class Node(object):
    def __init__(self,
                 name, parent, children, ancestors, siblings, depth, sub_hierarchy, idx):
        self.name = name
        self.parent = parent
        self.depth = depth
        self.children = children
        self.ancestors = ancestors
        self.siblings = siblings
        self.sub_hierarchy = sub_hierarchy
        self.idx = idx

        # for plotting
        self.HCL = (0., 0., 0.)
        self.hue_range = (0., 0.)
        self.hex = ''

class Tree(object):
    def __init__(self, i2c, json):

        self.i2c = i2c
        self.K = max([i for i, c in self.i2c.items()]) + 1
        self.json = json
        if len(self.json) == 0:
            self.json = {'root': {c: {} for _, c in self.i2c.items()}}
        self.root = 'root'
        self.target_classes = np.array([i for i, _ in self.i2c.items()])
        self.c2i = {c:i for i, c in self.i2c.items()}
        self.nodes = self.init_nodes()
        self.train_class = list(self.i2n.keys())
        self.init_matrices()
        self.nodes = colour_nodes(self.nodes, self.root)
        self.init_graph()
        self.init_metric_families()

        self.abs2con = self.bulid_depth_nodes()

        # fig, ax = plt.subplots(figsize=(100, 80))
        #
        # # 假设 tree 是已构建的层次结构对象，show_idx 是目标节点索引
        # plot_hierarchy(self, ax, show_idx=[i for i, _ in self.i2n.items()])
        #
        # # 显示图形
        # plt.show()




    def init_nodes(self):

        idx_counter = self.K
        self.M = self.K
        self.i2n = self.i2c.copy()
        q = queue.Queue()
        nodes = {}

        root_node = Node(name=self.root,
                          parent=None,
                          children=list(self.json[self.root].keys()),
                          ancestors=[],
                          siblings=[],
                          depth=0,
                          sub_hierarchy=self.json[self.root],
                          idx=-1)
        nodes[self.root] = root_node
        q.put(root_node)
        while not q.empty():
            parent = q.get()
            for c in parent.children:
                if c in self.c2i:
                    idx = self.c2i[c]
                else:
                    idx = idx_counter
                    idx_counter += 1
                child_node = Node(
                    name=c,
                    parent=parent.name,
                    children=list(parent.sub_hierarchy[c].keys()),
                    ancestors=parent.ancestors + [parent.name],
                    siblings=parent.children,
                    depth=parent.depth + 1,
                    sub_hierarchy=parent.sub_hierarchy[c],
                    idx=idx
                )
                if idx not in self.i2n:
                    self.i2n[idx] = c
                    self.M += 1

                nodes[c] = child_node
                q.put(child_node)
        self.n2i = {n:i for i, n in self.i2n.items()}

        return nodes
    def init_matrices(self):
        self.hmat = np.zeros((self.M, self.M), dtype=np.float32)
        self.sibmat = np.zeros((self.M, self.M), dtype=np.float32)
        for i in range(self.M):
            concept = self.i2n[i]
            sib_idx = [self.n2i[s] for s in self.nodes[concept].siblings]
            self.sibmat[i] = np.array([i in sib_idx for i  in range(self.M)]).astype(np.float32)
            chid_idx = [i] + [self.n2i[c] for c in self.nodes[concept].ancestors if c != self.root]
            self.hmat[i] = np.array([i in chid_idx for i in range(self.M)]).astype(np.float32)

        self.sibmat = torch.tensor(self.sibmat, dtype=torch.float32, requires_grad=False)
        self.hmat = torch.tensor(self.hmat, dtype=torch.float32, requires_grad=False)

    def init_graph(self, ):
        """ Initializes networkx graph, used for visualization. """
        rels = json2rels(self.json)
        self.G = nx.Graph()
        self.G.add_edges_from(rels)
        pos = hierarchy_pos(self.G, self.root, width=2 * math.pi)
        self.pos = {
            u: (r * math.cos(theta), r * math.sin(theta)) for u, (theta, r) in pos.items()
        }
    def is_hyponym_of(self, key, target):
        if self.nodes[key].parent is None:
            return False
        if self.nodes[key].parent == target:
            return True
        else:
            return self.is_hyponym_of(self.nodes[key].parent, target)

    def metric_family(self, concept):
        node = self.nodes[concept]
        siblings = [i for i in self.target_classes if self.is_hyponym_of(self.i2c[i], node.parent)]
        cousins = [i for i in self.target_classes if
                   self.is_hyponym_of(self.i2c[i], self.nodes[node.parent].parent)]
        return siblings, cousins

    def init_metric_families(self, ):
        for i in self.target_classes:
            name = self.i2c[i]
            node = self.nodes[name]

            metric_siblings, metric_cousins = self.metric_family(name)
            if node.parent != 'root':
                node.metric_siblings = metric_siblings
            else:
                # parent is root, no hierarchical relaxation as it would include all nodes
                node.metric_siblings = [i]
                node.metric_cousins = [i]
                continue

            if self.nodes[node.parent].parent != 'root':
                # we know the parent is not root if we are here
                node.metric_cousins = metric_cousins
            else:
                node.metric_cousins = metric_siblings

    def search_leaves(self, node):
        nodes = []

        if len(node.children) == 0:
            return [node.idx]
        else:
            for name in node.children:
                nodes = nodes + self.search_leaves(self.nodes[name])

        return nodes

    def depth_nodes(self):

        depth = {}
        for node in self.nodes.values():
            if node.depth == 0:
                continue
            if node.depth not in depth.keys():
                depth[node.depth] = [node.idx]
            else:
                depth[node.depth].append(node.idx)

        self.depth_idx = depth
        return depth

    def bulid_depth_nodes(self):
        pre_depth = []

        leave2depth = dict()
        depth_nodes = self.depth_nodes()
        for depth in depth_nodes.keys():
            idx_list = depth_nodes[depth]
            nodes_list = [self.nodes[self.i2n[idx]] for idx in idx_list]
            nodes_list = nodes_list + pre_depth
            pre_depth = []
            cur_depth = dict()
            for node in nodes_list:
                cur_depth.update({i: node.idx for i in self.search_leaves(node)})
                if len(node.children) == 0:
                    pre_depth.append(node)

            leave2depth[depth] = cur_depth
        return leave2depth




