import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# Load data
df = pd.read_csv('VRC01/VRC01_kds.csv')

print(df.head())

# epistasis heatmap (average binding for every combination of two mutations)
antigens = ['SF162', 'CH505TF']
sites = [col for col in df.columns if 'site_' in col]
n_sites = len(sites)

for antigen in antigens:
    df_antigen = df[df['antigen'] == antigen]

    data_hm = np.zeros((n_sites, n_sites))

    for i in range(n_sites):
        for j in range(n_sites):
            y_n_mutated = (df_antigen[sites[i]] != 'G') & (df_antigen[sites[j]] != 'G') # sees how many matured sites for each mutation
            data_hm[i, j] = df_antigen[y_n_mutated]['nlog10_Kd'].mean()
    sns.heatmap(data_hm, xticklabels=sites, yticklabels=sites)
    plt.title(f'Average affinity for {antigen} pairwise mutation')
    plt.show()
    
    
#epistasis differs between SF162 and CH505TF: scatter plot to compare interaction strength (with each antigen on an axis)


#higher order epistatsis (3)

