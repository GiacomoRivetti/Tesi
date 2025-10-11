import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
import time
import math

class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.neighbors = set()
        self.cluster_id = node_id

    def add_neighbor(self, neighbor_id):
        self.neighbors.add(neighbor_id)

class Cluster:
    def __init__(self, nodes):
        self.nodes = nodes
        self.ids = {n.id for n in nodes}
        self.min_id = min(self.ids)
        
        for node in nodes:
            node.cluster_id = self.min_id


    def sample_outgoing_edge(self, base_graph, cluster_map, max_attempts=10):
        outgoing_edges = []

        for node in self.nodes:
            for neighbor in base_graph[node.id]:
                if cluster_map.get(neighbor, self.min_id) != self.min_id:
                    outgoing_edges.append((node.id, neighbor))

        if not outgoing_edges:
            return None

        for _ in range(max_attempts):
            u, v = random.choice(outgoing_edges)
            if cluster_map.get(v, self.min_id) != self.min_id:
                return (u, v)

        return random.choice(outgoing_edges)

    
    def merge(self, other):
        merged_nodes = list(set(self.nodes + other.nodes))
        return Cluster(merged_nodes)
    
    
    def create_expander(self, log_degree=8, walk_length=5):
        cluster_nodes = {node.id for node in self.nodes}
        node_dict = {node.id: node for node in self.nodes}

        for node in self.nodes:
            node.new_neighbors = set([v for v in node.neighbors if v not in cluster_nodes])

        tokens = []
        for node in self.nodes:
            for _ in range(log_degree // 2):
                tokens.append({
                    "origin": node.id,
                    "current": node.id
                })

        for token in tokens:
            for _ in range(walk_length):
                neighs = [v for v in node_dict[token["current"]].neighbors if v in cluster_nodes]
                if not neighs:
                    break
                token["current"] = random.choice(neighs)

        arrivals = {}
        for token in tokens:
            arrivals.setdefault(token["current"], []).append(token)

        for dest, toks in arrivals.items():
            max_accept = max(1, 3 * log_degree // 8)
            accepted = random.sample(toks, min(len(toks), max_accept))
            for t in accepted:
                origin = node_dict[t["origin"]]
                dest_node = node_dict[dest]
                if origin.id != dest_node.id:
                    origin.new_neighbors.add(dest_node.id)
                    dest_node.new_neighbors.add(origin.id)

        for node in self.nodes:
            node.neighbors = node.new_neighbors
            del node.new_neighbors

    def degree_reduction(self, delta=4, walk_length=None, c=None):
        n = len(self.nodes)
        if n <= 1:
            return

        if walk_length is None:
            walk_length = int(math.log2(n)) * 3 
        if c is None:
            c = max(1, delta // 2)

        node_dict = {node.id: node for node in self.nodes}
        cluster_nodes = {node.id for node in self.nodes}

        for node in self.nodes:
            node.new_neighbors = set([v for v in node.neighbors if v not in cluster_nodes])

        tokens = []
        for node in self.nodes:
            for _ in range(c):
                tokens.append({"origin": node.id, "current": node.id, "active": True})

        for _ in range(int(math.log2(n))):
            new_tokens = []
            for token in tokens:
                if not token["active"]:
                    new_tokens.append(token)
                    continue
                current = token["current"]
                for _ in range(walk_length):
                    neighs = [v for v in node_dict[current].neighbors if v in cluster_nodes]
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
    def __init__(self, graph=None, num_nodes=None):
        self.results = []

        if graph is not None:
            self.base_graph = {u: list(graph.neighbors(u)) for u in graph.nodes()}

            self.nodes = [Node(i) for i in range(graph.number_of_nodes())]
            for u, v in graph.edges():
                self.nodes[u].add_neighbor(v)
                self.nodes[v].add_neighbor(u)
        else:
            self.nodes = [Node(i) for i in range(num_nodes)]
            self.initialize_network()
            self.base_graph = {node.id: list(node.neighbors) for node in self.nodes}

        self.clusters = [Cluster([n]) for n in self.nodes]

    def generate_sketch_matrix(self):
        np.random.seed(42)
        return np.random.choice([-1, 1], size=(self.sketch_size, 10000))
    
    def test_algorithm(graph, stages=20, runs=10):
        clusters_list = []
        internal_list = []
        time_list = []
        rounds_list = []
        init_cond_list = []
        final_cond_list = []
        init_diam_list = []
        final_diam_list = []

        for r in range(runs):
            init_cond = OverlayNetwork.min_conductance_pathlike(graph)
            init_diam = OverlayNetwork.diameter_or_lcc(graph)

            init_cond_list.append(init_cond)
            init_diam_list.append(init_diam)

            net = OverlayNetwork(graph=graph)
            net.run(stages=stages)

            if net.results:
                last = net.results[-1]
                clusters_list.append(last['clusters'])
                internal_list.append(last['internal_conductance'])
                time_list.append(last['time_sec'])
                rounds_list.append(len(net.results))

                G_final = nx.Graph()
                for node in net.nodes:
                    for neigh in node.neighbors:
                        G_final.add_edge(node.id, neigh)

                final_cond = OverlayNetwork.min_conductance_pathlike(G_final)
                final_diam = OverlayNetwork.diameter_or_lcc(G_final)

                final_cond_list.append(final_cond)
                final_diam_list.append(final_diam)

        stats = {
            'clusters': {
                'mean': float(np.mean(clusters_list)),
                'std': float(np.std(clusters_list))
            },
            'internal_conductance_final': {
                'mean': float(np.mean(internal_list)),
                'std': float(np.std(internal_list))
            },
            'init_cond': {
                'mean': float(np.mean(init_cond_list)),
                'std': float(np.std(init_cond_list))
            },
            'final_cond': {
                'mean': float(np.mean(final_cond_list)),
                'std': float(np.std(final_cond_list))
            },
            'init_diameter': {
                'mean': float(np.mean(init_diam_list)),
                'std': float(np.std(init_diam_list))
            },
            'final_diameter': {
                'mean': float(np.mean(final_diam_list)),
                'std': float(np.std(final_diam_list))
            },
            'time_sec': {
                'mean': float(np.mean(time_list)),
                'std': float(np.std(time_list))
            },
            'rounds': {
                'mean': float(np.mean(rounds_list)),
                'std': float(np.std(rounds_list))
            }
        }
        return stats

    def run(self, stages=5):
        with open('results.csv', 'w', newline='') as csvfile:
            fieldnames = ['stage', 'clusters', 'internal_conductance', 'external_conductance', 'time_sec']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for stage in range(stages):
                start_time = time.time()
                
                if len(self.clusters) <= 1:
                    break

                new_clusters = []
                merged = set()
                all_cluster_ids = {c.min_id for c in self.clusters}

                random.shuffle(self.clusters)
                n = len(self.nodes)
                attempts_per_cluster = max(1, int(math.log2(n)))
                cluster_map = {n.id: c.min_id for c in self.clusters for n in c.nodes}

                i = 0
                while i < len(self.clusters):
                    if self.clusters[i].min_id in merged:
                        i += 1
                        continue

                    current = self.clusters[i]

                    chosen_edge = None
                    for _ in range(attempts_per_cluster):
                        cluster_map = {n.id: c.min_id for c in self.clusters for n in c.nodes}
                        edge = current.sample_outgoing_edge(self.base_graph, cluster_map)
                        if edge is not None:
                            chosen_edge = edge
                            break

                    if chosen_edge is None:
                        new_clusters.append(current)
                        merged.add(current.min_id)
                        i += 1
                        continue

                    u, v = chosen_edge
                    other_node = v if u in current.ids else u
                    target_cluster = None
                    for j in range(len(self.clusters)):
                        if j == i: 
                            continue
                        if other_node in self.clusters[j].ids:
                            target_cluster = self.clusters[j]
                            break

                    if target_cluster and target_cluster.min_id not in merged:
                        merged_cluster = current.merge(target_cluster)
                        new_clusters.append(merged_cluster)
                        merged.update([current.min_id, target_cluster.min_id])
                        i += 1
                    else:
                        new_clusters.append(current)
                        merged.add(current.min_id)
                        i += 1

                self.clusters = new_clusters
                for c in self.clusters:
                    c.create_expander()
                    c.degree_reduction(delta=16)

                conductance = self.estimate_conductance()
                elapsed_time = time.time() - start_time
                internal_phi = 0.0
                if self.clusters:
                    vals = [OverlayNetwork.internal_conductance(c.nodes, samples=30) for c in self.clusters]
                    internal_phi = min(vals) if vals else 0.0

                print(f"Stage {stage+1}: clusters={len(self.clusters)}, "
                      f"external conductance={conductance:.4f}, "
                      f"internal conductance={internal_phi:.4f}, "
                      f"time={elapsed_time:.2f}s")
                writer.writerow({
                    'stage': stage + 1,
                    'clusters': len(self.clusters),
                    'internal_conductance': internal_phi,
                    'external_conductance': conductance,
                    'time_sec': elapsed_time
                })
                self.results.append({
                    'stage': stage + 1,
                    'clusters': len(self.clusters),
                    'internal_conductance': internal_phi,
                    'external_conductance': conductance,
                    'time_sec': elapsed_time
                })

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
    
    def internal_conductance(nodes, samples=50):
        G = nx.Graph()
        for node in nodes:
            for neigh in node.neighbors:
                G.add_edge(node.id, neigh)

        n = len(G.nodes)
        if n <= 1:
            return 0.0

        phi_values = []
        for _ in range(samples):

            k = random.randint(1, n-1)
            S = set(random.sample(list(G.nodes), k))
            phi = nx.algorithms.cuts.conductance(G, S)
            phi_values.append(phi)

        return min(phi_values) if phi_values else 0.0
    
    def min_conductance_pathlike(G):
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

        nodes = list(G.nodes())
        n = len(nodes)
        min_phi = float("inf")

        for k in range(1, n):
            S = set(nodes[:k])
            phi = nx.algorithms.cuts.conductance(G, S)
            if phi < min_phi:
                min_phi = phi

        return min_phi
    
    def diameter_or_lcc(G):
        if G.number_of_nodes() == 0:
            return 0
        if nx.is_connected(G):
            return nx.diameter(G)
        lcc = max(nx.connected_components(G), key=len)
        return nx.diameter(G.subgraph(lcc).copy())


    
if __name__ == "__main__":
    graph_type = "BA"  # ER, BA, WS, SBM, REGULAR

    if graph_type == "ER":
        G = nx.erdos_renyi_graph(n=1024, p=0.009)

    elif graph_type == "BA":
        G = nx.path_graph(1024)

    elif graph_type == "WS":
        G = nx.watts_strogatz_graph(n=1024, k=6, p=0.1)

    elif graph_type == "REGULAR":
        G = nx.random_regular_graph(d=4, n=1024)

    elif graph_type == "SBM":
        sizes = [100, 100, 100]
        probs = [
            [0.8, 0.02, 0.01],
            [0.02, 0.7, 0.02],
            [0.01, 0.02, 0.9]
            ]
        G = nx.stochastic_block_model(sizes, probs)

    else:
        raise ValueError("Tipo di grafo non supportato")

    stats = OverlayNetwork.test_algorithm(G, stages=20, runs=2)
    print("Risultati medi su 10 run:")
    for metric, vals in stats.items():

        print(f"{metric}: media={vals['mean']:.4f}, std={vals['std']:.4f}")
