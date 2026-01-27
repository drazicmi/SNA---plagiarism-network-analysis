import pandas as pd
import numpy as np
import networkx as nx

filePath = 'data-set/Results 1_anonymized.csv'

# SECTION_1: Dataset analysis

# Load dataset and display data
data = pd.read_csv(filePath)
print(data)

# Check for empty fields
print(f'Are there empty fields?\n{data.isna().any()}')

# Calculate number of students
unique_students_df = pd.concat([data['acter1'], data['acter2']]).to_frame().drop_duplicates()
num_of_students = unique_students_df.shape[0]
print(f'Total number of students: {num_of_students}')

# Calculate occurrences for each student
print(f'Are student occurrences in pairs unique? Answer: { data["acter1"].is_unique }')  # False

student_Occurrence = pd.concat([data['acter1'], data['acter2']]).to_frame('acter')

student_Occurrence = student_Occurrence.groupby('acter')
# Note that agg(np.size) and size() do the same, in future callables such as np.size will be replaced with function calls
print(student_Occurrence['acter'].agg(np.size).sort_values(ascending=False))

# Calculate total number of similarity lines
student1_lines = data[['acter1', 'lines_matched']].rename(columns={'acter1': 'acter'})
student2_lines = data[['acter2', 'lines_matched']].rename(columns={'acter2': 'acter'})

student_similarity = pd.concat([student1_lines[['acter', 'lines_matched']], student2_lines[['acter', 'lines_matched']]])
student_similarity = student_similarity.groupby('acter')
# Avoided agg(np.sum) and swaped it for sum() because of the runtime warning
print(student_similarity.sum().sort_values(by='lines_matched', ascending=False))


# SECTION_2 : Research questions

"""
    Start by creating a undirected graph using networkX.
    Graph is undirected.
    Nodes represent students, Edges represent connection between two students.
"""


graph = nx.Graph()

# First add each student as a node (44 unique students)
graph.add_nodes_from(unique_students_df[0])

# Now we add edges (if a row that connects two students exists, we add it as an edge)
set_edges = set()
for _, acter1, acter2 in data[['acter1', 'acter2']].itertuples():
    set_edges.add((acter1, acter2))
graph.add_edges_from(set_edges)

# print(graph) -> Graph with 44 nodes and 250 edges

# Question 1
network_density = nx.density(graph)
print(f'Network density: {network_density}')

# Question 2
# Calculate average_shortest_path_length and network diameter
average_shortest_path_length = nx.average_shortest_path_length(graph)
print(f'Average shortest path length: {average_shortest_path_length}')

diameter = nx.diameter(graph)
print(f'Diameter: {diameter}')

# Question 3
# We know graph isn't fully connected, because of the network density (it's not 1)
# We can calculate number of connections and the length for each component

num_of_connections = nx.number_connected_components(graph)
print(f'Number of connected components: {num_of_connections}')

