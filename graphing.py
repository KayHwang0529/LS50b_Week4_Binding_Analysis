#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

#1.2
# Load dataå
df = pd.read_csv('VRC01/VRC01_kds.csv')
print(df.head())  # Display the first few rows of the dataframe to verify loading

# Compute number of germline residues
df['n_germline'] = df['vh_aa'].apply(lambda x: x.count('G'))

# Pivot data for replicate comparison
df_pivot = df.pivot_table(index='id', columns='replicate', values='nlog10_Kd', aggfunc='first')
df_pivot.columns = [f'replicate_{int(col)}' for col in df_pivot.columns]

#1.3b
# Scatter plot of replicate 1 vs replicate 2
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pivot, x='replicate_1', y='replicate_2', alpha=0.6)
plt.plot([df_pivot['replicate_1'].min(), df_pivot['replicate_1'].max()],
         [df_pivot['replicate_1'].min(), df_pivot['replicate_1'].max()],
         color='red', linestyle='--')  # 1:1 line
plt.title('Scatter Plot of nlog10Kd: Replicate 1 vs Replicate 2')
plt.xlabel('nlog10Kd Replicate 1')
plt.ylabel('nlog10Kd Replicate 2')
plt.show()

# INTERPRETATION for Scatter Plot of nlog10Kd: Replicate 1 vs Replicate 2
# Most points lie close to the 1:1 line, indicating that the replicates generally agree with each other.
# There are a few outliers, and this could possibly represent noise.


# Compute Pearson and Spearman correlation coefficients
pearson_corr, pearson_p = stats.pearsonr(df_pivot['replicate_1'].dropna(), df_pivot['replicate_2'].dropna())
spearman_corr, spearman_p = stats.spearmanr(df_pivot['replicate_1'].dropna(), df_pivot['replicate_2'].dropna())

# Replicate-replicate plots for each antigen with consistent axis ranges
# Calculate global min and max across all antigens for consistent axis ranges
all_replicate_values = []
for antigen in df['antigen'].unique():
    antigen_data = df[df['antigen'] == antigen]
    antigen_pivot = antigen_data.pivot_table(index='id', columns='replicate', values='nlog10_Kd', aggfunc='first')
    antigen_pivot.columns = [f'replicate_{int(col)}' for col in antigen_pivot.columns]
    all_replicate_values.extend(antigen_pivot['replicate_1'].dropna().values)
    all_replicate_values.extend(antigen_pivot['replicate_2'].dropna().values)

global_min = np.nanmin(all_replicate_values)
global_max = np.nanmax(all_replicate_values)

# Create replicate-replicate plots for each antigen
for antigen in df['antigen'].unique():
    antigen_data = df[df['antigen'] == antigen]
    antigen_pivot = antigen_data.pivot_table(index='id', columns='replicate', values='nlog10_Kd', aggfunc='first')
    antigen_pivot.columns = [f'replicate_{int(col)}' for col in antigen_pivot.columns]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=antigen_pivot, x='replicate_1', y='replicate_2', alpha=0.6)
    plt.plot([global_min, global_max], [global_min, global_max], color='red', linestyle='--')  # 1:1 line
    plt.title(f'Scatter Plot of nlog10Kd: Replicate 1 vs Replicate 2 ({antigen})')
    plt.xlabel('nlog10Kd Replicate 1')
    plt.ylabel('nlog10Kd Replicate 2')
    plt.show()
    
    # Compute correlations for each antigen
    corr_data_r1 = antigen_pivot['replicate_1'].dropna()
    corr_data_r2 = antigen_pivot['replicate_2'].dropna()
    # Keep only overlapping indices to ensure that we are comparing the same variants
    overlap_idx = corr_data_r1.index.intersection(corr_data_r2.index)
    if len(overlap_idx) > 1:
        pearson_corr_antigen, pearson_p_antigen = stats.pearsonr(corr_data_r1[overlap_idx], corr_data_r2[overlap_idx])
        spearman_corr_antigen, spearman_p_antigen = stats.spearmanr(corr_data_r1[overlap_idx], corr_data_r2[overlap_idx])
        print(f'{antigen} - Pearson correlation: {pearson_corr_antigen:.4f}, p-value: {pearson_p_antigen:.2e}')
        print(f'{antigen} - Spearman correlation: {spearman_corr_antigen:.4f}, p-value: {spearman_p_antigen:.2e}')

# Overall correlations
pearson_corr, pearson_p = stats.pearsonr(df_pivot['replicate_1'].dropna(), df_pivot['replicate_2'].dropna())
spearman_corr, spearman_p = stats.spearmanr(df_pivot['replicate_1'].dropna(), df_pivot['replicate_2'].dropna())

print(f'Overall Pearson correlation: {pearson_corr:.4f}, p-value: {pearson_p:.2e}')
print(f'Overall Spearman correlation: {spearman_corr:.4f}, p-value: {spearman_p:.2e}')

# INTERPRETATION for Scatter Plot of nlog10Kd: Replicate 1 vs Replicate 2 (CH505TF)
# CH505TF points are generally more widely distributed and not as tightly-clustered around the 1:1 line, suggesting more variability between replicate 1 and replicate 2's measurements.

# INTERPRETATION for Scatter Plot of nlog10Kd: Replicate 1 vs Replicate 2 (SF162)
# SF162 points are generally more tightly clustered, indicating good agreement between replicate 1 and replicate 2's measurements.

#1.3c
# Histograms of nlog10Kd
plt.figure(figsize=(10, 8))
sns.histplot(df['nlog10_Kd'], bins=30, kde=True, color='blue', alpha=0.4)
sns.histplot(df['nlog10_Kd'], bins=20, kde=True, color='red', alpha=0.5)
sns.histplot(df['nlog10_Kd'], bins=10, kde=True, color='green', alpha = 0.6)
plt.title('Distribution of nlog10Kd')
plt.xlabel('nlog10Kd')
plt.ylabel('Frequency')
plt.show()

# INTERPRETATION for Distribution of nlog10Kd
# nlog10Kd values are generally unimodal and centered around a value of 7.0.
# There's a slight right tail, which represents binders with a higher relative affinity.

# Separate data by antigen
for antigen in df['antigen'].unique():
    antigen_data = df[df['antigen'] == antigen]
    plt.figure(figsize=(10, 8))
    sns.histplot(antigen_data['nlog10_Kd'], bins=30, kde=True, color='blue', alpha=0.4)
    sns.histplot(antigen_data['nlog10_Kd'], bins=20, kde=True, color='red', alpha=0.5)
    sns.histplot(antigen_data['nlog10_Kd'], bins=10, kde=True, color='green', alpha=0.6)
    plt.title(f'Distribution of nlog10_Kd for {antigen}')
    plt.xlabel('nlog10_Kd')
    plt.ylabel('Frequency')
    plt.show()

# INTERPRETATION for Distribution of nlog10_Kd for CH505TF
# nlog10_Kd distribution is wider and centered around lower values, indicating more variability and weaker binding affinity in general.
# The distribution is weighted towards the left, which suggests that there might be a lot of variants with weak binding capabilities.

# INTERPRETATION for Distribution of nlog10_Kd for SF162
# nlog10_Kd distribution is tighter and centered around 7.0-7.4, indicating consistent and a relatively stronger binding affinity.
# There is a bit of a right tail with suggests that very few variants reach an even higher binding affinity beyond the distribution peak nlog10_Kd value.


# Load metadata to determine germline vs mature status
metadata = pd.read_csv('VRC01/VRC01_metadata.csv')
germline_aa = metadata['germline_aa'].tolist()
mature_aa = metadata['mature_aa'].tolist()

# Function to classify variant as germline-like or mature-like
def classify_variant(row):
    site_values = [row[f'site_{i}'] for i in range(10)]
    # Count matches to germline and mature
    germline_matches = sum(1 for i in range(10) if site_values[i] == germline_aa[i])
    mature_matches = sum(1 for i in range(10) if site_values[i] == mature_aa[i])
    # Classify based on which is closer
    if germline_matches >= mature_matches:
        return 'Germline-like'
    else:
        return 'Mature-like'

df['variant_type'] = df.apply(classify_variant, axis=1)

# Calculate mean nlog10_Kd across replicates for each variant and antigen combination
df_mean = df.groupby(['id', 'antigen', 'variant_type', 'n_germline']).agg({
    'nlog10_Kd': 'mean'
}).reset_index()

#1.3d
# Box and Violin Plots for variant type by antigen (using mean Kd across replicates)
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_mean, x='antigen', y='nlog10_Kd', hue='variant_type')
plt.title('Box Plots of Mean nlog10_Kd by Antigen and Germline/Mature Identity')
plt.ylabel('Mean nlog10_Kd (across replicates)')
plt.show()

# INTERPRETATION for Box Plots of Mean nlog10_Kd by Antigen and Germline/Mature Identity
# Mature-like variants have a slightly higher median of nlog10_Kd for both CH505TF and SF162, suggesting that binding generally improves after maturation.
# SF162 variants generally have a higher binding affinity than CH505TF variants.


plt.figure(figsize=(12, 6))
sns.violinplot(data=df_mean, x='antigen', y='nlog10_Kd', hue='variant_type', split=False)
plt.title('Violin Plots of Mean nlog10_Kd by Antigen and Germline/Mature Identity')
plt.ylabel('Mean nlog10_Kd (across replicates)')
plt.show()

# INTERPRETATION for Violin Plots of Mean nlog10_Kd by Antigen and Germline/Mature Identity
# Mature-like variants generally have slightly higher binding affinities than germline-like variants, suggesting that binding generally improves after maturation.
# For both antigen types, individual variants vary substantially in terms of binding affinity regardless if mature-like or germ-like.


#1.3e
#Box and Violin Plots for n_germline vs nlog10_Kd
plt.figure(figsize=(12, 8))
sns.boxplot(data=df_mean, x='n_germline', y='nlog10_Kd', hue='antigen', dodge=True)
plt.title('Box Plot of Mean nlog10_Kd by Number of Germline Residues')
plt.xlabel('Number of Germline Residues')
plt.ylabel('Mean nlog10_Kd (across replicates)')
plt.show()

# INTERPRETATION for Box Plot of Mean nlog10_Kd by Number of Germline Residues
# Variants with 11 germline residues generally have high median average nlog10_Kd values compared to those with 12/13 germline residues, demonstrating how more mutated antibodies might display better binding.
# SF162 has higher median values than CH505TF, indicating that there is generally always an advantage to binding to SF162 across varying levels of germline residues.


plt.figure(figsize=(12, 8))
sns.violinplot(data=df_mean, x='n_germline', y='nlog10_Kd', hue='antigen', split=True)
plt.title('Violin Plot of Mean nlog10_Kd by Number of Germline Residues')
plt.xlabel('Number of Germline Residues')
plt.ylabel('Mean nlog10_Kd (across replicates)')
plt.show()

# INTERPRETATION for Violin Plot of Mean nlog10_Kd by Number of Germline Residues
# Antibodies with the same number of germline residues still vary substantially in binding affinity (wide vertical spread).
# SF162 parts of the violin plots are consistently higher relative to the CH505TF plots, suggesting that SF162 is generally bound more strongly across varying levels of germline residues.


#1.3f
# Scatter plot for SF162 vs CH505TF using mean nlog10_Kd across replicates
df_sf162_mean = df_mean[df_mean['antigen'] == 'SF162'][['id', 'nlog10_Kd', 'n_germline']].rename(
    columns={'nlog10_Kd': 'nlog10_Kd_SF162'}
)
df_ch505_mean = df_mean[df_mean['antigen'] == 'CH505TF'][['id', 'nlog10_Kd']].rename(
    columns={'nlog10_Kd': 'nlog10_Kd_CH505TF'}
)

merged = pd.merge(df_sf162_mean, df_ch505_mean, on='id', how='inner')
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=merged,
    x='nlog10_Kd_SF162',
    y='nlog10_Kd_CH505TF',
    hue='n_germline',
    palette='viridis',
    alpha=0.6
)
plt.title('Scatter Plot of Mean nlog10_Kd: SF162 vs CH505TF')
plt.xlabel('Mean nlog10_Kd for SF162 (across replicates)')
plt.ylabel('Mean nlog10_Kd for CH505TF (across replicates)')
plt.grid(True)
plt.show()

# INTERPRETATION FOR Scatter Plot of Mean nlog10_Kd: SF162 vs CH505TF
# There is a clear upward-slope of data points, suggesting a positive relationship between average nlog10Kd values across CH505TF and SF162 variants.
# Germline content influences binding affinity, but colored data points show no clear pattern, suggesting that germline content doesn't create preferences for one antigen over the other.


#1.4
# Summary statistics for nlog10_Kd
summary_stats = df['nlog10_Kd'].describe()

# Mean, median, standard deviation, and IQR
mean_kd = summary_stats['mean']
median_kd = summary_stats['50%']
std_kd = summary_stats['std']
q1_kd = summary_stats['25%']
q3_kd = summary_stats['75%']

print(f'Mean nlog10_Kd: {mean_kd}')
print(f'Median nlog10_Kd: {median_kd}')
print(f'Standard Deviation nlog10_Kd: {std_kd}')
print(f'IQR nlog10_Kd: {q3_kd - q1_kd}')

# Summary statistics by antigen
summary_by_antigen = df.groupby('antigen')['nlog10_Kd'].describe()
print('Summary Statistics by Antigen:')
print(summary_by_antigen)


#1.5
# Check for missing data
missing_data = df.isnull().sum()
print('Missing Data Count:')
print(missing_data)

# Check for unique variants
unique_variants = df['id'].nunique()
print(f'Unique Variants Count: {unique_variants}')

# Check for outliers using IQR method
Q1 = df['nlog10_Kd'].quantile(0.25)
Q3 = df['nlog10_Kd'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['nlog10_Kd'] < lower_bound) | (df['nlog10_Kd'] > upper_bound)]
print(f'Outliers Count: {outliers.shape[0]}')

# Check for potential data duplication
duplicates = df[df.duplicated()]
print(f'Duplicates Count: {duplicates.shape[0]}')

#2.0 
# Use replicate-averaged values and align by antigen
sf162_for_test = df_mean[df_mean['antigen'] == 'SF162'][['id', 'nlog10_Kd', 'n_germline']].rename(
    columns={'nlog10_Kd': 'nlog10_Kd_SF162'}
)
ch505_for_test = df_mean[df_mean['antigen'] == 'CH505TF'][['id', 'nlog10_Kd']].rename(
    columns={'nlog10_Kd': 'nlog10_Kd_CH505TF'}
)

perm_df = pd.merge(sf162_for_test, ch505_for_test, on='id', how='inner')


x = perm_df['nlog10_Kd_SF162'].to_numpy()
y = perm_df['nlog10_Kd_CH505TF'].to_numpy()


def mean_diff_stat(a, b):
    return np.mean(a) - np.mean(b)

observed_mean_diff = mean_diff_stat(x, y)
mean_diff_perm = stats.permutation_test(
    data=(x,y),
    statistic=mean_diff_stat,
    permutation_type='samples',
    n_resamples=10000,
    alternative='two-sided',
    random_state=50
)

plt.figure(figsize=(10, 6))
plt.hist(mean_diff_perm.null_distribution, bins=50, color='blue', alpha=0.75, edgecolor='black')
plt.axvline(observed_mean_diff, color='red', linestyle='--', linewidth=2,
            label=f'Observed difference = {observed_mean_diff:.4f}')
plt.title('Null Distribution (Permutation): Mean Difference')
plt.xlabel('Difference of means (mean nlog10_Kd)')
plt.ylabel('Count')
plt.legend()
plt.show()

print(f'Observed effect size (SF162 - CH505TF): {observed_mean_diff:.4f}')
print(f'Permutation p-value: {mean_diff_perm.pvalue:.4e}')
if mean_diff_perm.pvalue < 0.05:
    direction = 'higher' if observed_mean_diff > 0 else 'lower'
    print(f'Interpretation: reject H0. Mean -log10KD for SF162 is {direction} than CH505TF.')
else:
    print('Interpretation: fail to reject H0. No strong evidence of a mean difference between antigens.')


observed_pearson = stats.pearsonr(x, y)[0]
observed_spearman = stats.spearmanr(x, y)[0]

pearson_perm = stats.permutation_test(
    data=(x, y),
    statistic=lambda a, b: stats.pearsonr(a, b)[0],
    permutation_type='pairings',
    n_resamples=10000,
    alternative='two-sided',
    random_state=50
)

spearman_perm = stats.permutation_test(
    data=(x, y),
    statistic=lambda a, b: stats.spearmanr(a, b)[0],
    permutation_type='pairings',
    n_resamples=10000,
    alternative='two-sided',
    random_state=50
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(pearson_perm.null_distribution, bins=50, color='purple', alpha=0.75, edgecolor='black')
axes[0].axvline(observed_pearson, color='red', linestyle='--', linewidth=2,
                label=f'Observed Pearson r = {observed_pearson:.4f}')
axes[0].set_title('Null Distribution (Permutation): Pearson r')
axes[0].set_xlabel('Pearson r')
axes[0].set_ylabel('Count')
axes[0].legend()

axes[1].hist(spearman_perm.null_distribution, bins=50, color='green', alpha=0.75, edgecolor='black')
axes[1].axvline(observed_spearman, color='red', linestyle='--', linewidth=2,
                label=f'Observed Spearman ρ = {observed_spearman:.4f}')
axes[1].set_title('Null Distribution (Permutation): Spearman ρ')
axes[1].set_xlabel('Spearman ρ')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f'Observed Pearson r: {observed_pearson:.4f}')
print(f'Pearson permutation p-value: {pearson_perm.pvalue:.4e}')
if pearson_perm.pvalue < 0.05:
    print('Interpretation (Pearson): reject H0.')
else:
    print('Interpretation (Pearson): fail to reject H0.')

print(f'Observed Spearman ρ: {observed_spearman:.4f}')
print(f'Spearman permutation p-value: {spearman_perm.pvalue:.4e}')
if spearman_perm.pvalue < 0.05:
    print('Interpretation (Spearman): reject H0.')
else:
    print('Interpretation (Spearman): fail to reject H0.')

# PART 3

# analysis #1: identifying sites that influencing binding affinity the most
important_sites = []
for site in range(10):
    column = f'site_{site}'
    for antigen in df['antigen'].unique():
        unique_antigen = df[df['antigen'] == antigen]

        g_aa = unique_antigen[column].mode().iloc[0] # most frequent location for the double A germline

        g_vals = unique_antigen[unique_antigen[column] == g_aa]['nlog10_Kd']
        m_vals = unique_antigen[unique_antigen[column] != g_aa]['nlog10_Kd']
        if len(m_vals) < 20 or len(g_vals) < 20: # makes sure there's enough mutants
            continue

        difference = m_vals.mean() - g_vals.mean() # average change of affinity for each site
        important_sites.append({'site': site, 'antigen': antigen, 'mean_g': g_vals.mean(), 'mean_m': m_vals.mean(), 'difference': difference})
important_df = pd.DataFrame(important_sites)
print(important_df.sort_values('difference')) # from this can see that sites 7 and 4 have largest negative differences, sites

# INTERPRETATION
# Comparing the mean -logKd values for G/M sequences for each antigen show that mutations at sites 4 and 7 result in the greatest differences in affinity for both antigen types.
# This suggests that sites 4 and 7 are crucial for maintaining strong binding affinities for the antibodies because mutating them results in weaker binding affinities.
# Additionally, sites 6, 8 and 9 show significant positive differences.
# This means that mutations at these sites generally increase -logKd.

# analysis #2: epistasis between two sites of interest (4 and 7)



# analysis #3: how number of G/M sites affects correlations between SF162 and CH505TF
correlations = []

for count_g_sites in sorted(merged['n_germline'].unique()):
    variants = merged[merged['n_germline'] == count_g_sites]
    if len(variants) < 10:
        continue
    r_p, _ = stats.pearsonr(variants["nlog10_Kd_SF162"], variants["nlog10_Kd_CH505TF"])
    correlations.append({"n_germline": count_g_sites, "pearson_r": r_p})
correlations_df = pd.DataFrame(correlations)

sns.barplot(correlations_df, x='n_germline', y='pearson_r')
sns.barplot(correlations_df, x='n_germline', y='pearson_r')
plt.xlabel('Number of germline residues')
plt.ylabel('Pearson r value for SF162 vs CH505TF')
plt.title('Correlation between SF162 and CH505TF versus antigen maturation level')
plt.show()

# INTERPRETATION for Correlation between SF162 and CH505TF versus antigen maturation level
# Across all maturation levels, SF162 and CH505TF are strongly correlated (Pearson r ranges from around 0.7 to 0.8), which suggests that variants that bind to one antigen generally bind strongly to the other antigen.
# As antibodies have more germline residues, the correlation value decreases slightly. This suggests that germline variants may display more antigen-specific variation.

# four panels

# %%
