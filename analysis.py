import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

# Load data
df = pd.read_csv('VRC01/VRC01_kds.csv')

print(df.head())

# epistasis heatmap (average binding for every combination of two mutations)
# higher # = better binding, lighter color = higher affinity (b/c -log values)
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
    
    
# most notably between sites 4 and 7

# mutation count vs. affinity
# are there any cutoffs of mutation #s (fitness valleys) that result in worse binding?
df['n_mutations'] = (df[sites] != 'G').sum(axis=1)
sns.lineplot(data=df, x='n_mutations', y='nlog10_Kd', hue='antigen')

plt.xlabel('Number of mutations')
plt.ylabel('Binding affinity (-logKd)')
plt.title('Binding affinity vs. number of mutations for SF162 and CH505TF')
plt.show()
# higher y-values = more fit (better binding to antigen)
# CH505TF has to mutate/mature much more to achieve same level of binding affinity as SF162
# generally both have smooth fitness landscapes/evolution

# at 1 mutation, affinity for CH505TF dips a bit but SF162 doesn't
# could be a fitness valley? maybe CH505TF makes 2 mutations at once to skip the valley
# more difficult to get to 2 mutations than just 1
# CH505TF steeper slope between 2-6 mutations --> each additional mutation = more binding power compared to SF162

    
#epistasis differs between SF162 and CH505TF: scatter plot to compare interaction strength (with each antigen on an axis)


#higher order epistatsis (3)
