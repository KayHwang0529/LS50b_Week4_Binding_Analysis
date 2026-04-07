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

    
#Epistasis differs between SF162 and CH505TF: scatter plot to compare interaction strength (extent of epistatsis)

# Extract interaction strengths (epistasis) for each site pair
interaction_data = []

for i in range(n_sites):
    for j in range(i+1, n_sites):  # Only upper triangle to avoid duplicates
        # SF162
        df_sf162 = df[df['antigen'] == 'SF162']
        both_mut_sf162 = (df_sf162[sites[i]] != 'G') & (df_sf162[sites[j]] != 'G')
        neither_mut_sf162 = (df_sf162[sites[i]] == 'G') & (df_sf162[sites[j]] == 'G')
        only_i_sf162 = (df_sf162[sites[i]] != 'G') & (df_sf162[sites[j]] == 'G')
        only_j_sf162 = (df_sf162[sites[i]] == 'G') & (df_sf162[sites[j]] != 'G')
        
        # Calculate mean binding for each combination
        both_val_sf162 = df_sf162[both_mut_sf162]['nlog10_Kd'].mean()
        neither_val_sf162 = df_sf162[neither_mut_sf162]['nlog10_Kd'].mean()
        i_val_sf162 = df_sf162[only_i_sf162]['nlog10_Kd'].mean()
        j_val_sf162 = df_sf162[only_j_sf162]['nlog10_Kd'].mean()
        
        # Epistasis = actual - expected additive
        expected_sf162 = (i_val_sf162 - neither_val_sf162) + (j_val_sf162 - neither_val_sf162)
        actual_sf162 = both_val_sf162 - neither_val_sf162
        epistasis_sf162 = actual_sf162 - expected_sf162
        
        # repeat for CH505TF
        df_ch505 = df[df['antigen'] == 'CH505TF']
        both_mut_ch505 = (df_ch505[sites[i]] != 'G') & (df_ch505[sites[j]] != 'G')
        neither_mut_ch505 = (df_ch505[sites[i]] == 'G') & (df_ch505[sites[j]] == 'G')
        only_i_ch505 = (df_ch505[sites[i]] != 'G') & (df_ch505[sites[j]] == 'G')
        only_j_ch505 = (df_ch505[sites[i]] == 'G') & (df_ch505[sites[j]] != 'G')
        
        # Calculate mean binding for each combination
        both_val_ch505 = df_ch505[both_mut_ch505]['nlog10_Kd'].mean()
        neither_val_ch505 = df_ch505[neither_mut_ch505]['nlog10_Kd'].mean()
        i_val_ch505 = df_ch505[only_i_ch505]['nlog10_Kd'].mean()
        j_val_ch505 = df_ch505[only_j_ch505]['nlog10_Kd'].mean()
        
        # Epistasis = actual - expected additive
        expected_ch505 = (i_val_ch505 - neither_val_ch505) + (j_val_ch505 - neither_val_ch505)
        actual_ch505 = both_val_ch505 - neither_val_ch505
        epistasis_ch505 = actual_ch505 - expected_ch505
        
        interaction_data.append({
            'sites': f'{sites[i]}-{sites[j]}',
            'sf162': epistasis_sf162,
            'ch505tf': epistasis_ch505
        })

interaction_df = pd.DataFrame(interaction_data)

# Add a column to identify which antigen has stronger epistasis
interaction_df['stronger_antigen'] = interaction_df.apply(
    lambda row: 'SF162' if row['sf162'] > row['ch505tf'] else 'CH505TF', axis=1
)

# Scatter plot with color coding by antigen
plt.figure(figsize=(8, 8))
colors = {'SF162': '#1f77b4', 'CH505TF': '#ff7f0e'}
for antigen, color in colors.items():
    mask = interaction_df['stronger_antigen'] == antigen
    plt.scatter(interaction_df[mask]['sf162'], interaction_df[mask]['ch505tf'], 
                s=100, alpha=0.6, label=antigen, color=color)

# Add diagonal line for reference
min_val = min(interaction_df['sf162'].min(), interaction_df['ch505tf'].min())
max_val = max(interaction_df['sf162'].max(), interaction_df['ch505tf'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

plt.xlabel('Epistasis (SF162 interaction strength)')
plt.ylabel('Epistasis (CH505TF interaction strength)')
plt.title('Epistatic interactions differ between SF162 and CH505TF')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
plt.grid(True, alpha=0.3)
plt.legend(title='Stronger epistasis', loc='upper left')
plt.show()

# Findings:
# Points above diagonal = strong epistasis in CH505TF but weak in SF162 (antigen-specific interactions)
# Points below diagonal = strong epistasis in SF162 but weak in CH505TF
# Scatter from diagonal indicates that genetic background (antigen) shapes how mutations interact
# This suggests epistasis is not universal but antigen-dependent, thus with the goal of breadth not potency, epistasis may not be the best feature to tune
# Epistasis of CH505TF is also more variable than SF162, which has more consistent interaction strengths across site pairs (more points clustered around diagonal)
# this further explains the fitness valleys in CH505TF and the more rugged landscape, as some mutations may have strong positive interactions while others have strong negative interactions

#higher order epistatsis (3)






# lab meeting 3

# for epistasis plot, want to see exactly which sites work with which sites to see the most positive and negative relationships
plt.figure(figsize=(8, 8))
colors = {'SF162': '#1f77b4', 'CH505TF': '#ff7f0e'}
for antigen, color in colors.items():
    mask = interaction_df['stronger_antigen'] == antigen
    plt.scatter(interaction_df[mask]['sf162'], interaction_df[mask]['ch505tf'], 
                s=100, alpha=0.6, label=antigen, color=color)

plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)

plt.xlabel('Epistasis (SF162 interaction strength)')
plt.ylabel('Epistasis (CH505TF interaction strength)')
plt.title('Epistatic interactions differ between SF162 and CH505TF (labeled sites with |epistasis coeff| > 0.05)')
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
plt.grid(True, alpha=0.3)
plt.legend(title='Stronger epistasis', loc='upper left')
# added below from above code (last week) to label the site relationships
for i, row in interaction_df.iterrows():
    if abs(row['sf162']) > 0.05 and abs(row['ch505tf']) > 0.05: # picked this cutoff based on graph to select outliers
        plt.annotate(row['sites'], (row['sf162'], row['ch505tf']))
plt.show()

# want a plot to compare sf162 vs ch505tf affinity to analyze broadness, not just how th

