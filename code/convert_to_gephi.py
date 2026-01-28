import pandas as pd
import os

filePath2 = os.path.join(os.path.dirname(__file__), 'data-set', 'Results 2_anonymized.csv')
df2 = pd.read_csv(filePath2)

gephi_df2 = pd.DataFrame({
    'Source': df2['acter1'],
    'Target': df2['acter2'],
    'Type': 'Undirected',
    'Weight': df2['lines_matched']
})

outputPath2 = os.path.join(os.path.dirname(__file__), 'data-set', 'Results_2_Gephi.csv')
gephi_df2.to_csv(outputPath2, index=False)

filePath1 = os.path.join(os.path.dirname(__file__), 'data-set', 'Results 1_anonymized.csv')
df1 = pd.read_csv(filePath1)

gephi_df1 = pd.DataFrame({
    'Source': df1['acter1'],
    'Target': df1['acter2'],
    'Type': 'Undirected',
    'Weight': df1['lines_matched']
})

outputPath1 = os.path.join(os.path.dirname(__file__), 'data-set', 'Results_1_Gephi.csv')
gephi_df1.to_csv(outputPath1, index=False)

