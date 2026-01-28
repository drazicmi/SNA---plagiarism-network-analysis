
import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
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

# Question 4

N = graph.number_of_nodes()
p = graph.number_of_edges()
G_er = nx.erdos_renyi_graph(N, p/(N*(N-1)/2))
G_ba = nx.barabasi_albert_graph(N, p // N)

avg_clust_er = nx.average_clustering(G_er)
print(f"ER graph clustering coefficient: {avg_clust_er}")
avg_clust_ba = nx.average_clustering(G_ba)
print(f"BA graph clustering coefficient: {avg_clust_ba}")
avg_clust = nx.average_clustering(graph)
print(f'Average clustering coefficient: {avg_clust}\n')

global_clust_er = nx.transitivity(G_er)
print(f'Global clustering for ER graph: {global_clust_er}')
global_clust_ba = nx.transitivity(G_ba)
print(f'Global clustering for BA graph: {global_clust_ba}')
global_clust = nx.transitivity(graph)
print(f'Global coefficient: {global_clust}')

clustering = nx.clustering(graph)

plt.hist(list(clustering.values()), bins=10)
plt.title("Clustering Coefficient Distribution")
plt.show()

avg_clust = nx.average_clustering(graph)

# Question 5

# High local clustering coefficient and low average 
# shortest path length indicate small-world properties.

# Question 6

r = nx.degree_assortativity_coefficient(graph)
print(f'Degree assortativity coefficient: {r}')

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
    heuristic[v] = (
        0.3 * deg_n[v] +
        0.2 * close_n[v] +
        0.2 * between_n[v] +
        0.3 * eig_n[v]
    )

top5 = sorted(heuristic.items(), key=lambda x: x[1], reverse=True)[:5]
print("\nTop 5 nodes by heuristic centrality:")
print(top5)
