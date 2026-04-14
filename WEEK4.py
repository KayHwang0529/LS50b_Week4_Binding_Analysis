# week 4: 4.13.26

import pandas as pd

# took pymol distances between pairwise sites and min distances between each site and closest gp120 residue
# turned into file to see if there are relationships between distance + epistatic coefficients
distances = pd.read_csv('/Users/katehsia/Desktop/all_distances.csv')
print(distances)

# separate csv file rows into the pairwise site distances and distances from site to gp120 residues
pairwise_d = distances.iloc[0:4]
gp120_pair_d = distances.iloc[4:8]

# create a dictionary and turn that into a dataframe with the distances to compare with epistasis coefficients
# note that we only care about the pairwise distances here
distance_data = {
    'sites': ['site_4-site_6', 'site_5-site_6', 'site_5-site_8', 'site_5-site_9'],
    'distance': [7.0, 3.8, 9.1, 11.9]
}
distance_df = pd.DataFrame(distance_data)

# pulling epistasis coefficients for relevant sites
epistasis_pairs = interaction_df[interaction_df['sites'].isin(distance_data['sites'])]
print(epistasis_pairs)

# combine epistasis coefficients with distances for plotting (match w/ sites and keep coefficients on the left)
all_data = epistasis_pairs.merge(distance_df, on='sites', how='left')
print(all_data)

# PLOT IT!!
plt.scatter(all_data['distance'], all_data['sf162'], label='SF162')
plt.scatter(all_data['distance'], all_data['ch505tf'], label='CH505TF')
plt.xlabel('Distance between selected site pairs (Å)')
plt.ylabel('Pairwise epistasis coefficients between selected sites')
plt.title('Epistasis coefficients vs distances for selected site pairs')
plt.legend()
plt.show()

# X-AXIS
# distance between pairwise sites in Å

# Y-AXIS
# positive epistasis coefficient = SYNERGISTIC (double mutant binds better than individual mutants)
# negative espistasis coefficient = ANTAGONISTIC (double mutant makes binding worse)

# in these four pairs, short distances do not guarantee strong/positive epistasis (in fact, seem antigen-specific)
# almost as if for extreme distances, SF162 prefers longer distances while CH505TF prefers shorter distances
# epistasis coefficient signs can flip between antigens for a given distance --> antigen-specific epistasis
# POSSIBLE CONCLUSION: epistasis shaped by local structure/antigen-specific constraints, not just residue contact
# strong positive and negative epistasis occurs at ~7-9Å (mid-range distances)
# OVERALL no monotonic relationship (closer != better epistasis)

# for antigen-specific:
# distances measured are fixed
# SF162 and CH505TF different around env binding sites
# double mutating --> changes shape/orientation
# can be beneficial for one antigen's binding site but not the other's
# glycan positioning (to expose/hide binding site, push loops into different conformations)
# + preferred conformations around binding site, mutations have different effects on
# structure/energy of each env site

# NEXT STEPS: look at physical structures of SF162/CH505TF and their CD4 binding sites
# research more about possible qualitative reasons to explain antigen-specific epistasis