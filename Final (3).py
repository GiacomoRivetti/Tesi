import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import time
import math

class Node:
    def __init__(self, node_id, sketch_size):
        self.id = node_id
        self.neighbors = set()
        self.sketch_vector = np.zeros(sketch_size)
        self.local_sketch = None
        self.random_bits = None
        self.cluster_id = node_id

    def add_neighbor(self, neighbor_id):
        self.neighbors.add(neighbor_id)

    def compute_edge_id(self, neighbor_id):
        u, v = sorted((self.id, neighbor_id))
        return hash((u, v)) % 10000

    def compute_sketch(self, MG):
        self.sketch_vector = np.zeros(MG.shape[0])
        for neighbor in self.neighbors:
            edge_id = self.compute_edge_id(neighbor)
            self.sketch_vector += MG[:, edge_id]

class Cluster:
    def __init__(self, nodes, sketch_size):
        self.nodes = nodes
        self.ids = {n.id for n in nodes}
        self.min_id = min(self.ids)
        self.sketch_size = sketch_size
        self.random_bits = self.generate_random_bits()
        
        for node in nodes:
            node.cluster_id = self.min_id
            node.random_bits = self.random_bits

    def generate_random_bits(self, length=16):
        return ''.join(random.choice(['0', '1']) for _ in range(length))

    def aggregate_sketch(self, MG, gossip_rounds=10):
        for node in self.nodes:
            node.compute_sketch(MG)

        states = {node.id: (node.sketch_vector.copy(), 1.0 if node.id == self.min_id else 0.0)
                  for node in self.nodes}

        for _ in range(gossip_rounds):
            new_states = {nid: (np.zeros(self.sketch_size), 0.0) for nid in states}
            for nid, (sketch, weight) in states.items():
                targets = [nid, random.choice(list(self.ids))]
                for t in targets:
                    new_states[t] = (new_states[t][0] + sketch / 2,
                                     new_states[t][1] + weight / 2)
            states = new_states

        total_sketch = sum(s for (s, w) in states.values())
        return total_sketch

    def sample_outgoing_edge(self, sketch_agg, all_cluster_ids):
        outgoing_edges = []
        for node in self.nodes:
            for neighbor in node.neighbors:
                if neighbor not in self.ids:
                    outgoing_edges.append((node.id, neighbor))

        if not outgoing_edges:
            return None
        return random.choice(outgoing_edges)

    def find_outgoing_edges(self, all_cluster_ids):
        outgoing = []
        for node in self.nodes:
            for neighbor in node.neighbors:
                if neighbor not in self.ids:
                    outgoing.append((node.id, neighbor))
        return outgoing
    
    def merge(self, other):
        merged_nodes = list(set(self.nodes + other.nodes))
        return Cluster(merged_nodes, self.sketch_size)
    
    def create_expander(self, degree_factor=20):
        n = len(self.nodes)
        if n <= 1:
            return

        for node in self.nodes:
            node.neighbors = set()

        k = max(1, degree_factor * int(math.log2(n)))

        node_ids = [node.id for node in self.nodes]
        for node in self.nodes:
            choices = random.sample(node_ids, min(k, n-1))
            for target_id in choices:
                if target_id != node.id:
                    node.neighbors.add(target_id)
                    target_node = next(nd for nd in self.nodes if nd.id == target_id)
                    target_node.neighbors.add(node.id)

    def degree_reduction(self, delta=3, walk_length=None, c=None):

        n = len(self.nodes)
        if n <= 1:
            return

        if walk_length is None:
            walk_length = int(math.log2(n))
        if c is None:
            c = max(1, delta // 4)

        node_dict = {node.id: node for node in self.nodes}

        for node in self.nodes:
            node.new_neighbors = set()

        tokens = []
        for node in self.nodes:
            for _ in range(c):
                tokens.append({
                    "origin": node.id,
                    "current": node.id,
                    "active": True
                })

        for _ in range(int(math.log2(n))):
            new_tokens = []
            for token in tokens:
                if not token["active"]:
                    new_tokens.append(token)
                    continue

                current = token["current"]
                for _ in range(walk_length):
                    neighs = list(node_dict[current].neighbors)
                    if not neighs:
                        break
                    current = random.choice(neighs)

                token["current"] = current
                new_tokens.append(token)

            loc_map = {}
            for token in new_tokens:
                if token["active"]:
                    loc_map.setdefault(token["current"], []).append(token)

            for node_id, toks in loc_map.items():
                if len(toks) <= delta:
                    for t in toks:
                        t["active"] = False

            tokens = new_tokens

        for token in tokens:
            if not token["active"]:
                origin = node_dict[token["origin"]]
                dest = node_dict[token["current"]]
                if origin.id != dest.id:
                    origin.new_neighbors.add(dest.id)
                    dest.new_neighbors.add(origin.id)

        for node in self.nodes:
            node.neighbors = node.new_neighbors
            del node.new_neighbors

class OverlayNetwork:
    def __init__(self, num_nodes, sketch_size=64):
        self.sketch_size = sketch_size
        self.nodes = [Node(i, sketch_size) for i in range(num_nodes)]
        self.initialize_network()
        self.clusters = [Cluster([n], sketch_size) for n in self.nodes]
        self.results = []

    def initialize_network(self):
        """Create a random initial network topology"""
        G = nx.erdos_renyi_graph(len(self.nodes), 0.1)
        for u, v in G.edges():
            self.nodes[u].add_neighbor(v)
            self.nodes[v].add_neighbor(u)

    def generate_sketch_matrix(self):
        np.random.seed(42)
        return np.random.choice([-1, 1], size=(self.sketch_size, 10000))

    def run(self, stages=5):
        with open('results.csv', 'w', newline='') as csvfile:
            fieldnames = ['stage', 'clusters', 'conductance', 'time_sec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for stage in range(stages):
                start_time = time.time()
                
                if len(self.clusters) <= 1:
                    break

                MG = self.generate_sketch_matrix()
                new_clusters = []
                merged = set()
                all_cluster_ids = {c.min_id for c in self.clusters}

                random.shuffle(self.clusters)

                i = 0
                while i < len(self.clusters):
                    if self.clusters[i].min_id in merged:
                        i += 1
                        continue

                    current = self.clusters[i]
                    sketch_agg = current.aggregate_sketch(MG)
                    edge = current.sample_outgoing_edge(sketch_agg, all_cluster_ids)

                    if edge is None:
                        new_clusters.append(current)
                        merged.add(current.min_id)
                        i += 1
                        continue

                    u, v = edge
                    other_node = v if u in current.ids else u
                    target_cluster = None
                    for j in range(i+1, len(self.clusters)):
                        if other_node in self.clusters[j].ids:
                            target_cluster = self.clusters[j]
                            break

                    if target_cluster and target_cluster.min_id not in merged:
                        merged_cluster = current.merge(target_cluster)
                        new_clusters.append(merged_cluster)
                        merged.update([current.min_id, target_cluster.min_id])
                        i += 2
                    else:
                        new_clusters.append(current)
                        merged.add(current.min_id)
                        i += 1

                self.clusters = new_clusters
                merged_cluster.create_expander()
                merged_cluster.degree_reduction(delta=8)
                conductance = self.estimate_conductance()
                elapsed_time = time.time() - start_time

                # Write stage results to CSV
                writer.writerow({
                    'stage': stage + 1,
                    'clusters': len(self.clusters),
                    'conductance': conductance,
                    'time_sec': round(elapsed_time, 2)
                })

                print(f"Stage {stage+1}: clusters={len(self.clusters)}, conductance={conductance:.4f}, time={elapsed_time:.2f}s")


    def estimate_conductance(self):
        if len(self.clusters) <= 1:
            return 0.0

        G = nx.Graph()
        for node in self.nodes:
            for neighbor in node.neighbors:
                G.add_edge(node.id, neighbor)

        min_conductance = float('inf')
        for cluster in self.clusters:
            S = {n.id for n in cluster.nodes}
            if not S or len(S) == len(G.nodes()):
                continue

            cut_size = nx.cut_size(G, S)
            volume = sum(d for n, d in G.degree(S))
            if volume == 0:
                continue

            conductance = cut_size / min(volume, 2*len(G.edges()) - volume)
            min_conductance = min(min_conductance, conductance)

        return min_conductance if min_conductance != float('inf') else 0.0

if __name__ == "__main__":
    network = OverlayNetwork(num_nodes=256, sketch_size=1024)
    network.run(stages=30)