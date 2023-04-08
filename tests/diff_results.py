

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Files
expected = 'qmec_expected_1962_2019.txt'
output = 'qmec_output_1962_2019.txt'

# Load data into dataframes
df_expected = pd.read_csv(expected, sep = ' ', names=['Day_exp', 'Hour_exp', 'Q_exp'], dtype={'Day_exp': str, 'Hour_exp': str, 'Q_exp': float})
df_output = pd.read_csv(output, sep = ' ', names=['Day_out', 'Hour_out', 'Q_out'], dtype={'Day_out': str, 'Hour_out': str, 'Q_out': float})

# Combine dataframes
df = pd.concat([df_expected, df_output], axis=1)

# Compute difference between expected results and output
df['delta'] = df.apply(lambda row: np.absolute(row['Q_out'] - row['Q_exp']), axis=1)

print(f"Maximum delta : {df['delta'].max()} in row {df['delta'].idxmax()}")
print(f"Mean delta : {df['delta'].mean()}")

# df['delta_group'] = pd.cut(df.delta, bins=range(0, 180, 10), right=False)
# (df.groupby('delta_group').delta.value_counts()
#    .unstack().plot.bar(width=1, stacked=True))

# df.groupby('delta').size().plot.bar(width=1)
df.hist(column=['delta'],bins=200, figsize=(8,6))

# df['delta'].plot(kind = 'kde')
plt.show()

df.to_csv('differences.csv', index=False)
