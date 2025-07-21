import os
import glob

# Check what files are in the processed directory
processed_dir = "/Users/chideraangwadom/serine-protease-ligands/gnn-catalysis-predictor/data/processed"
print("Files in processed directory:")
for file in os.listdir(processed_dir):
    if file.endswith('.csv'):
        print(f"  📄 {file}")

# Search for the files in the broader data directory
data_dir = "/Users/chideraangwadom/serine-protease-ligands/gnn-catalysis-predictor/data/raw"
print(f"\nSearching for CSV files in {data_dir}:")
for file in glob.glob(f"{data_dir}/**/*kcat*.csv", recursive=True):
    print(f"  📄 {file}")
for file in glob.glob(f"{data_dir}/**/*km*.csv", recursive=True):
    print(f"  📄 {file}")

import pandas as pd

def merge_and_analyze_efficiency(
    kcat_file: str = "/Users/chideraangwadom/serine-protease-ligands/gnn-catalysis-predictor/data/raw/filtered_kcat_serine_proteases.csv",
    km_file: str = "/Users/chideraangwadom/serine-protease-ligands/gnn-catalysis-predictor/data/raw/filtered_km_serine_proteases.csv",
    merged_outfile: str = "/Users/chideraangwadom/serine-protease-ligands/gnn-catalysis-predictor/data/processed/merged_kcat_km_data.csv",
    correlation_outfile: str = "/Users/chideraangwadom/serine-protease-ligands/gnn-catalysis-predictor/data/processed/log10_efficiency_correlations.csv",
    correlation_method: str = "pearson"
):
    """
    Merge kcat and km data, calculate efficiency, and analyze correlations.
    """
    try:
        # Read input files
        kcat_df = pd.read_csv(kcat_file)
        km_df = pd.read_csv(km_file)
        
        # Merge dataframes
        merged_df = pd.merge(
            kcat_df, km_df,
            on=['sequence', 'sequence_source', 'uniprot'],
            suffixes=('_kcat', '_km')
        )
        
        # Calculate log10 efficiency
        merged_df['log10_efficiency'] = (
            merged_df['log10kcat_max'] - merged_df['log10km_mean']
        )
        
        # Define correlation features
        corr_features = [
            "log10_efficiency",
            "log10kcat_max", 
            "log10km_mean",
            "value_kcat",
            "value_km",
            "reaction_mw_diff_perc",
            "ph_kcat",
            "ph_km",
            "taxonomy_id_kcat"
        ]
        
        # Calculate correlations
        corr_df = merged_df[corr_features].dropna()
        correlation = corr_df.corr(method=correlation_method)
        
        # Save results
        correlation["log10_efficiency"].sort_values(ascending=False).to_csv(
            correlation_outfile
        )
        merged_df.to_csv(merged_outfile, index=False)
        
        print(f"✔ Merged data saved to: {merged_outfile}")
        print(f"✔ Correlation results saved to: {correlation_outfile}")
        
        return correlation["log10_efficiency"].sort_values(ascending=False)
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find input file - {e}")
        return None
    except KeyError as e:
        print(f"❌ Error: Missing expected column - {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

# Execute the function
if __name__ == "__main__":
    result = merge_and_analyze_efficiency()
    if result is not None:
        print("\n📊 Top correlations with log10_efficiency:")
        print(result.head())