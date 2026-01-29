
import networkx as nx
from networkx.algorithms.community import girvan_newman
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from networkx.linalg.laplacianmatrix import laplacian_matrix
filePath = os.path.join(os.path.dirname(__file__), 'data-set', 'Results 2_anonymized.csv')

data = pd.read_csv(filePath)

# Dataset
print(f'Data:\n', data.head(5))

# Is dataset valid?

acter1_is_int = data["acter1_percentage"].dtype == "int64"
acter2_is_int = data["acter2_percentage"].dtype == "int64"
lines_matched_is_int = data["lines_matched"].dtype == "int64"

acter1_in_range = data[(data["acter1_percentage"] < 0) | (data["acter1_percentage"] > 100)].empty
acter2_in_range = data[(data["acter2_percentage"] < 0) | (data["acter2_percentage"] > 100)].empty

lines_matched_non_negative = data[ data["lines_matched"] < 0 ].empty

print(f'Are there empty fields?\n{data.isna().any()}')
print(f'Lines matched are ints and non-negative? {lines_matched_is_int and lines_matched_non_negative}')
print(f'Acter percentages are ints and between 0-100? {acter1_is_int and acter2_is_int and acter1_in_range and acter2_in_range }')

# Dataset analysis

unique_students_df = pd.concat([data['acter1'], data['acter2']]).to_frame().drop_duplicates()

graph = nx.Graph()
graph.add_nodes_from(unique_students_df[0])
set_edges = set()
for _, acter1, acter2 in data[['acter1', 'acter2']].itertuples():
    set_edges.add((acter1, acter2))
graph.add_edges_from(set_edges)
print(f'Graph created. Number of nodes: {graph.number_of_nodes()}, number of edges: {graph.number_of_edges()}')

nx.draw(graph, with_labels=True)
plt.show()

# Question 1
d = nx.density(graph)
print(f'Network density: {d}')

# Question 2

avg_dist = nx.average_shortest_path_length(graph)
print(f'Average shortest path length: {avg_dist}')

diameter = nx.diameter(graph)
print(f'Network diameter: {diameter}')

# Question 3

print("Graph is connected?", nx.is_connected(graph))

deg_centrality = nx.degree_centrality(graph)
closeness_centrality = nx.closeness_centrality(graph)
betweenness_centrality = nx.betweenness_centrality(graph)
def degree_centralization(graph):
    degrees = dict(graph.degree())
    max_deg = max(degrees.values())
    n = len(degrees)

    return sum(max_deg - d for d in degrees.values()) / ((n - 1) * (n - 2))

C_real = degree_centralization(graph)
print("Centralization:", round(C_real, 3))

# Question 4

N = graph.number_of_nodes()
p = graph.number_of_edges()

trials = 10
er_clusts = []
er_trans = []
ba_clusts = []
ba_trans = []

for i in range(trials):

    prob = p / (N * (N - 1) / 2) if N > 1 else 0
    G_er = nx.erdos_renyi_graph(N, prob)
    er_clusts.append(nx.average_clustering(G_er))
    er_trans.append(nx.transitivity(G_er))

    m = max(1, int(p // N))
    m = min(m, N - 1) if N > 1 else 1
    G_ba = nx.barabasi_albert_graph(N, m)
    ba_clusts.append(nx.average_clustering(G_ba))
    ba_trans.append(nx.transitivity(G_ba))

avg_clust_er = sum(er_clusts) / trials
avg_clust_ba = sum(ba_clusts) / trials
avg_trans_er = sum(er_trans) / trials
avg_trans_ba = sum(ba_trans) / trials

print(f"ER graph average clustering coefficient (over {trials} trials): {avg_clust_er}")
print(f"BA graph average clustering coefficient (over {trials} trials): {avg_clust_ba}")
avg_clust = nx.average_clustering(graph)
print(f'Average clustering coefficient: {avg_clust}\n')

print(f'ER graph average transitivity (over {trials} trials): {avg_trans_er}')
print(f'BA graph average transitivity (over {trials} trials): {avg_trans_ba}')
global_clust = nx.transitivity(graph)
print(f'Global coefficient: {global_clust}')

clustering = nx.clustering(graph)

plt.hist(list(clustering.values()), bins=10)
plt.title("Raspodela lokalnih koeficijenata klasterizacije")
plt.show()

avg_clust = nx.average_clustering(graph)

# Question 5

L_real = nx.average_shortest_path_length(graph)
L_er = nx.average_shortest_path_length(G_er)
L_sf = nx.average_shortest_path_length(G_ba)

print(f'Average shortest path length ER: {L_er}')
print(f'Average shortest path length SF: {L_sf}')
print(f'Average shortest path length real: {L_real}')

# Question 6

r = nx.degree_assortativity_coefficient(graph)
print(f'\nDegree assortativity coefficient: {r}')

degrees = dict(graph.degree())

x = []
y = []

for u, v in graph.edges():
    x.append(degrees[u])
    y.append(degrees[v])

x = np.array(x)
y = np.array(y)

plt.figure(figsize=(6, 6))
plt.scatter(x, y, alpha=0.6, s=50)
plt.xlabel("Stepen čvora u")
plt.ylabel("Stepen čvora v")
plt.title("Asortativnost po stepenu čvorova na ivicama")
plt.grid(True)
plt.show()

# Question 7

degrees = [d for _, d in graph.degree()]

degree_count = Counter(degrees)
k = np.array(sorted(degree_count.keys()))
counts = np.array([degree_count[i] for i in k])

alpha = 2
k_fit = np.linspace(min(k), max(k), 200)

C = max(counts) * (min(k) ** alpha)
power_law = C * (k_fit ** (-alpha))

plt.figure(figsize=(6, 4))

plt.bar(k, counts, width=0.8, alpha=0.7, edgecolor='black',
        label="Empirijska raspodela stepena")

plt.plot(k_fit, power_law, 'r--',
         label=r"Idealizovana power-law raspodela ($\alpha = 2$)")

plt.xlabel("Stepen čvora k")
plt.ylabel("Broj čvorova")
plt.title("Raspodela čvorova po stepenu")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()

# Question 8

deg_centrality = {k: round(v, 2) for k, v in deg_centrality.items()}
closeness_centrality = {k: round(v, 2) for k, v in closeness_centrality.items()}
betweenness_centrality = {k: round(v, 2) for k, v in betweenness_centrality.items()}

def top_n(centrality_dict, n=5):
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]

print("Top 5 degree centrality:")
print(top_n(deg_centrality))

print("\nTop 5 closeness centrality:")
print(top_n(closeness_centrality))

print("\nTop 5 betweenness centrality:")
print(top_n(betweenness_centrality))

# Question 9

eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=1000)
eigenvector_centrality = {k: round(v, 2) for k, v in eigenvector_centrality.items()}

print("\nTop 5 eigenvector centrality:")
print(top_n(eigenvector_centrality))

# Question 10

def normalize(d):
    min_v = min(d.values())
    max_v = max(d.values())
    return {
        k: (v - min_v) / (max_v - min_v) if max_v > min_v else 0
        for k, v in d.items()
    }

deg_n = normalize(deg_centrality)
close_n = normalize(closeness_centrality)
between_n = normalize(betweenness_centrality)
eig_n = normalize(eigenvector_centrality)

heuristic = {}

for v in graph.nodes():
    heuristic[v] = round(
        0.166 * deg_n[v] +
        0.166 * close_n[v] +
        0.5 * between_n[v] +
        0.166 * eig_n[v],
        2
    )

top5 = sorted(heuristic.items(), key=lambda x: x[1], reverse=True)[:15]
print("\nTop 5 nodes by heuristic centrality:")
print(top5)

all_nodes = sorted(graph.nodes())

deg_vals = [deg_n[node] for node in all_nodes]
close_vals = [close_n[node] for node in all_nodes]
between_vals = [between_n[node] for node in all_nodes]
eig_vals = [eig_n[node] for node in all_nodes]

fig, ax = plt.subplots(figsize=(16, 6))

x = np.arange(len(all_nodes))
width = 0.2

bars1 = ax.bar(x - 1.5*width, deg_vals, width, label='Degree Centrality', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, close_vals, width, label='Closeness Centrality', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, between_vals, width, label='Betweenness Centrality', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, eig_vals, width, label='Eigenvector Centrality', alpha=0.8)

ax.set_xlabel('Čvorovi', fontsize=12)
ax.set_ylabel('Normalizovana vrednost', fontsize=12)
ax.set_title('Poređenje 4 centralnosti za sve čvorove', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(all_nodes, rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

#Question 13

graph2 = nx.Graph()
graph2.add_nodes_from(unique_students_df[0])
edges_with_weights = []

for _, row in data.iterrows():
    acter1 = row['acter1']
    acter2 = row['acter2']
    weight = row['lines_matched']
    edges_with_weights.append((acter1, acter2, weight))

graph2.add_weighted_edges_from(edges_with_weights)

L = laplacian_matrix(graph2).astype(float).todense()

eigenvalues = np.sort(np.real(np.linalg.eigvals(L)))

plt.figure(figsize=(8,5))
plt.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='o')
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.title("Spectral Analysis: Laplacian Eigenvalues")
plt.grid(True)
plt.show()

eigengaps = np.diff(eigenvalues)
predicted_communities = np.argmax(eigengaps) + 1

print("Eigenvalues:", eigenvalues)
print("Susedne razlike (eigengap):", eigengaps)
print(f"Predlog broja komuna na osnovu eigengap: {predicted_communities}")

num_connected_components = np.sum(np.isclose(eigenvalues, 0))
print(f"Broj povezanih komponenti (nulti eigenvalues): {num_connected_components}")


comp_gen = girvan_newman(graph2)

levels = []
for communities in comp_gen:
    level = [list(c) for c in communities]
    levels.append(level)
    if len(levels) >= graph2.number_of_nodes() - 1:  
        break

nodes = list(graph2.nodes())
n = len(nodes)
dist_matrix = np.zeros((n, n))

for level_idx, communities in enumerate(levels):
    for i, comm1 in enumerate(communities):
        for j, comm2 in enumerate(communities):
            if i >= j:
                continue
            for node1 in comm1:
                for node2 in comm2:
                    idx1 = nodes.index(node1)
                    idx2 = nodes.index(node2)
                   
                    if dist_matrix[idx1, idx2] == 0:
                        dist_matrix[idx1, idx2] = level_idx + 1
                        dist_matrix[idx2, idx1] = level_idx + 1

max_level = len(levels)
if max_level == 0:
    max_level = 1
for i in range(n):
    for j in range(i + 1, n):
        if dist_matrix[i, j] == 0:
            dist_matrix[i, j] = max_level + 1
            dist_matrix[j, i] = max_level + 1

condensed_dist = squareform(dist_matrix)

Z = linkage(condensed_dist, method='average')

plt.figure(figsize=(12, 6))
dendrogram(Z, labels=nodes, leaf_rotation=90)
plt.title("Girvan-Newman dendrogram")
plt.xlabel("Čvorovi")
plt.ylabel("Level of split")
plt.show()