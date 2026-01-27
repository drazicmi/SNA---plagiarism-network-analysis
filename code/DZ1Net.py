import pandas as pd
import numpy as np


filePath = 'data-set/Results 1_anonymized.csv'

# Load dataset and display data
data = pd.read_csv(filePath)
print(data)

# Calculate number of students
num_of_students = pd.concat([data['acter1'], data['acter2']]).drop_duplicates().shape[0]
print(f'Total number of students: {num_of_students}')

# Calculate occurrences for each student
print(f'Are student occurrences in pairs unique? Answer: { data["acter1"].is_unique }')

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