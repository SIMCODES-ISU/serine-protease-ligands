import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("filtered_km_serine_proteases.csv")

# Simulate material type
np.random.seed(42)
df['material_type'] = np.random.choice(['oil', 'solids'], size=len(df), p=[0.5, 0.5])
df['material_type_encoded'] = df['material_type'].map({'oil': 1.0, 'solids': 1.5})
df['material_type_noisy'] = df['material_type_encoded'] + np.random.normal(0, 0.05, size=len(df))

# Split into groups
df_oil = df[df['material_type'] == 'oil']
df_solids = df[df['material_type'] == 'solids']

# Plotting
plt.figure(figsize=(12, 4))

# 1. Histogram
plt.subplot(1, 3, 1)
plt.hist(df['log10km_mean'], bins=30, color='lightcoral', edgecolor='black')
plt.title('Histogram of log10km_mean')
plt.xlabel('log10km_mean')
plt.ylabel('Frequency')

# 2. Scatter plot
plt.subplot(1, 3, 2)
plt.scatter(df_oil['material_type_noisy'], df_oil['log10km_mean'], label='Oil', alpha=0.7)
plt.scatter(df_solids['material_type_noisy'], df_solids['log10km_mean'], label='Solids', alpha=0.7)
plt.title('Scatter: Material Type vs log10km_mean')
plt.xlabel('Material Type (noisy)')
plt.ylabel('log10km_mean')
plt.legend()

# 3. Cluster approximation using hexbin
plt.subplot(1, 3, 3)
plt.hexbin(df['material_type_noisy'], df['log10km_mean'], gridsize=30, cmap='viridis', alpha=0.7)
plt.title('Cluster Density Approximation')
plt.xlabel('Material Type (noisy)')
plt.ylabel('log10km_mean')
plt.colorbar(label='Density')

plt.tight_layout()
plt.show()
