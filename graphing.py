#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
# Load data
df = pd.read_csv('VRC01/VRC01_kds.csv')
print(df.head())  # Display the first few rows of the dataframe to verify loading

# Compute number of germline residues
df['n_germline'] = df['vh_aa'].apply(lambda x: x.count('G'))

# Pivot data for replicate comparison
df_pivot = df.pivot_table(index='id', columns='replicate', values='nlog10_Kd', aggfunc='first')
df_pivot.columns = [f'replicate_{int(col)}' for col in df_pivot.columns]

# Scatter plot of replicate 1 vs replicate 2
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pivot, x='replicate_1', y='replicate_2', alpha=0.6)
plt.plot([df_pivot['replicate_1'].min(), df_pivot['replicate_1'].max()],
         [df_pivot['replicate_1'].min(), df_pivot['replicate_1'].max()],
         color='red', linestyle='--')  # 1:1 line
plt.title('Scatter Plot of nlog10Kd: Replicate 1 vs Replicate 2')
plt.xlabel('nlog10Kd Replicate 1')
plt.ylabel('nlog10Kd Replicate 2')
plt.xlim(df_pivot['replicate_1'].min(), df_pivot['replicate_1'].max())
plt.ylim(df_pivot['replicate_2'].min(), df_pivot['replicate_2'].max())
plt.grid(True)
plt.show()

# Compute Pearson and Spearman correlation coefficients
pearson_corr, pearson_p = stats.pearsonr(df_pivot['replicate_1'].dropna(), df_pivot['replicate_2'].dropna())
spearman_corr, spearman_p = stats.spearmanr(df_pivot['replicate_1'].dropna(), df_pivot['replicate_2'].dropna())

print(f'Pearson correlation: {pearson_corr}, p-value: {pearson_p}')
print(f'Spearman correlation: {spearman_corr}, p-value: {spearman_p}')

# Histograms of nlog10Kd
plt.figure(figsize=(10, 8))
sns.histplot(df['nlog10_Kd'], bins=30, kde=True, color='blue', alpha=0.6)
plt.title('Distribution of nlog10Kd')
plt.xlabel('nlog10Kd')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# Separate data by antigen
for antigen in df['antigen'].unique():
    antigen_data = df[df['antigen'] == antigen]
    plt.figure(figsize=(10, 8))
    sns.histplot(antigen_data['nlog10_Kd'], bins=30, kde=True, alpha=0.6)
    plt.title(f'Distribution of nlog10_Kd for {antigen}')
    plt.xlabel('nlog10_Kd')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Box and Violin Plots for n_germline by antigen
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='antigen', y='nlog10_Kd', hue='replicate')
plt.title('Box Plots of nlog10_Kd by Antigen and Replicate')
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='antigen', y='nlog10_Kd', hue='replicate', split=True)
plt.title('Violin Plots of nlog10_Kd by Antigen and Replicate')
plt.show()

# Box and Violin Plots for n_germline vs nlog10_Kd
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='n_germline', y='nlog10_Kd', hue='antigen', dodge=True)
plt.title('Box Plot of nlog10_Kd by Number of Germline Residues')
plt.xlabel('Number of Germline Residues')
plt.ylabel('nlog10_Kd')
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
sns.violinplot(data=df, x='n_germline', y='nlog10_Kd', hue='antigen', split=True)
plt.title('Violin Plot of nlog10_Kd by Number of Germline Residues')
plt.xlabel('Number of Germline Residues')
plt.ylabel('nlog10_Kd')
plt.grid(True)
plt.show()

# Scatter plot for SF162 vs CH505TF by replicate
df_rep1_sf162 = df[(df['antigen'] == 'SF162') & (df['replicate'] == 1)].copy()
df_rep1_ch505 = df[(df['antigen'] == 'CH505TF') & (df['replicate'] == 1)].copy()

if len(df_rep1_sf162) > 0 and len(df_rep1_ch505) > 0:
    # Align by id for comparison
    merged = pd.merge(df_rep1_sf162[['id', 'nlog10_Kd', 'n_germline']], 
                      df_rep1_ch505[['id', 'nlog10_Kd']], 
                      on='id', suffixes=('_SF162', '_CH505'))
    if len(merged) > 0:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=merged, x='nlog10_Kd_SF162', y='nlog10_Kd_CH505', hue='n_germline', palette='viridis', alpha=0.6)
        plt.title('Scatter Plot of nlog10_Kd: SF162 vs CH505TF (Replicate 1)')
        plt.xlabel('nlog10_Kd SF162')
        plt.ylabel('nlog10_Kd CH505TF')
        plt.grid(True)
        plt.show()

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

# Data Quality Check
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
# %%
