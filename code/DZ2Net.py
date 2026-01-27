
import networkx as nx
import pandas as pd
import os
import matplotlib.pyplot as plt
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

#nx.draw(graph, with_labels=False)
#plt.show()

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
eigen_centrality = nx.eigenvector_centrality(graph, max_iter=1000)

# Question 4

N = graph.number_of_nodes()
p = graph.number_of_edges()
G_er = nx.erdos_renyi_graph(N, p)
#G_ba = nx.barabasi_albert_graph(N, p/N)

avg_clust_er = nx.average_clustering(G_er)
print(f"ER graph clustering coefficient: {avg_clust_er}")
#avg_clust_ba = nx.average_clustering(G_ba)
#print(f"BA graph clustering coefficient: {avg_clust_ba}")
avg_clust = nx.average_clustering(graph)
print(f'Average clustering coefficient: {avg_clust}')

global_clust_er = nx.transitivity(G_er)
print(f'Global clustering coefficient (transitivity) for ER graph: {global_clust_er}')
#global_clust_ba = nx.transitivity(G_ba)
#print(f'Global clustering coefficient (transitivity) for BA graph: {global_clust_ba}')
global_clust = nx.transitivity(graph)
print(f'Global clustering coefficient (transitivity): {global_clust}')

clustering = nx.clustering(graph)

plt.hist(list(clustering.values()), bins=10)
plt.title("Clustering Coefficient Distribution")
plt.show()

avg_clust = nx.average_clustering(graph)


