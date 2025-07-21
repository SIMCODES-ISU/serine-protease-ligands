import pandas as pd
import torch
import esm
from typing import List, Tuple, Optional, Dict
import re
import os
from tqdm import tqdm
import numpy as np

# Utility: filter non-standard amino acids
def is_valid_sequence(seq: str, valid_aa: set = set("ACDEFGHIKLMNPQRSTVWY")) -> bool:
    """Check if sequence contains only valid amino acids."""
    if not seq or not isinstance(seq, str):
        return False
    return all(residue.upper() in valid_aa for residue in seq.strip())

def validate_inputs(input_csv: str, sequence_column: str, id_column: Optional[str] = None) -> pd.DataFrame:
    """Validate and load CSV file with basic preprocessing."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")
    
    # Load CSV
    try:
        df = pd.read_csv(input_csv)
        print(f"📊 Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")
    
    # Check required columns
    if sequence_column not in df.columns:
        available_cols = ", ".join(df.columns.tolist())
        raise ValueError(f"Column '{sequence_column}' not found. Available columns: {available_cols}")
    
    if id_column and id_column not in df.columns:
        print(f"⚠️  ID column '{id_column}' not found. Will use auto-generated IDs.")
        id_column = None
    
    return df, id_column

def preprocess_sequences(
    df: pd.DataFrame, 
    sequence_column: str, 
    id_column: Optional[str] = None,
    filter_invalid: bool = True,
    max_length: Optional[int] = None
) -> List[Tuple[str, str]]:
    """Preprocess sequences: clean, deduplicate, filter."""
    
    initial_count = len(df)
    
    # Remove rows with missing sequences
    df = df.dropna(subset=[sequence_column])
    print(f"🧹 Removed {initial_count - len(df)} rows with missing sequences")
    
    # Clean sequences (remove whitespace, convert to uppercase)
    df[sequence_column] = df[sequence_column].astype(str).str.strip().str.upper()
    
    # Remove empty sequences
    df = df[df[sequence_column].str.len() > 0]
    print(f"🧹 Removed empty sequences. Remaining: {len(df)}")
    
    # Filter by length if specified
    if max_length:
        df = df[df[sequence_column].str.len() <= max_length]
        print(f"🧹 Filtered sequences longer than {max_length}. Remaining: {len(df)}")
    
    # Filter invalid amino acids
    if filter_invalid:
        before_filter = len(df)
        df = df[df[sequence_column].apply(is_valid_sequence)]
        filtered_count = before_filter - len(df)
        if filtered_count > 0:
            print(f"🧹 Filtered {filtered_count} sequences with invalid amino acids")
    
    # Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[sequence_column])
    dedup_count = before_dedup - len(df)
    if dedup_count > 0:
        print(f"🧹 Removed {dedup_count} duplicate sequences")
    
    # Create sequence pairs (id, sequence)
    if id_column and id_column in df.columns:
        sequences = list(zip(df[id_column].astype(str), df[sequence_column]))
    else:
        sequences = [(f"seq_{i:06d}", seq) for i, seq in enumerate(df[sequence_column])]
    
    print(f"✅ Final dataset: {len(sequences)} unique sequences")
    return sequences

def generate_esm2_embeddings_from_csv(
    input_csv: str = "merged_kcat_km_data.csv",
    output_file: str = "esm_embeddings.pt",
    sequence_column: str = "sequence",
    id_column: Optional[str] = None,
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 16,
    filter_invalid: bool = True,
    max_length: Optional[int] = None,
    device: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Generate ESM-2 embeddings from sequences in a CSV file.
    
    Args:
        input_csv: Path to input CSV file
        output_file: Path to save embeddings (.pt file)
        sequence_column: Name of column containing protein sequences
        id_column: Name of column containing sequence IDs (optional)
        model_name: ESM model to use
        batch_size: Batch size for processing
        filter_invalid: Remove sequences with non-standard amino acids
        max_length: Maximum sequence length to process
        device: Device to use ('cuda' or 'cpu', auto-detect if None)
    
    Returns:
        Dictionary mapping sequence IDs to embedding tensors
    """
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Validate inputs and load data
    df, id_column = validate_inputs(input_csv, sequence_column, id_column)
    
    # Preprocess sequences
    sequences = preprocess_sequences(
        df, sequence_column, id_column, filter_invalid, max_length
    )
    
    if len(sequences) == 0:
        raise ValueError("No valid sequences found after preprocessing!")
    
    # Load ESM model
    print(f"🔄 Loading ESM model: {model_name}")
    try:
        model, alphabet = getattr(esm.pretrained, model_name)()
        model = model.to(device)
        model.eval()
        batch_converter = alphabet.get_batch_converter()
        print(f"✅ Model loaded successfully")
    except Exception as e:
        raise ValueError(f"Error loading ESM model: {e}")
    
    # Generate embeddings in batches
    embeddings = {}
    total_batches = (len(sequences) + batch_size - 1) // batch_size
    
    print(f"🚀 Generating embeddings for {len(sequences)} sequences in {total_batches} batches...")
    
    try:
        for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
            batch_seqs = sequences[i:i + batch_size]
            
            # Convert sequences to tokens
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_seqs)
            batch_tokens = batch_tokens.to(device)
            
            # Generate embeddings
            with torch.no_grad():
                out = model(batch_tokens, repr_layers=[model.num_layers], return_contacts=False)
                reps = out["representations"][model.num_layers]
                
                # Extract per-sequence embeddings (mean over sequence length)
                for j, (label, seq) in enumerate(batch_seqs):
                    # Remove special tokens and average over sequence length
                    seq_len = len(seq)
                    emb = reps[j, 1:seq_len+1].mean(0).cpu()  # Shape: [embed_dim]
                    embeddings[label] = emb
    
    except Exception as e:
        print(f"❌ Error during embedding generation: {e}")
        raise
    
    # Save embeddings
    try:
        torch.save(embeddings, output_file)
        print(f"✅ Saved {len(embeddings)} embeddings to {output_file}")
        
        # Print some statistics
        embed_dim = next(iter(embeddings.values())).shape[0]
        file_size = os.path.getsize(output_file) / (1024**2)  # MB
        print(f"📊 Embedding dimension: {embed_dim}")
        print(f"📊 File size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error saving embeddings: {e}")
        raise
    
    return embeddings

def load_embeddings(filepath: str) -> Dict[str, torch.Tensor]:
    """Load embeddings from a .pt file."""
    try:
        embeddings = torch.load(filepath, map_location='cpu')
        print(f"📂 Loaded {len(embeddings)} embeddings from {filepath}")
        return embeddings
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        raise

def inspect_embeddings(embeddings: Dict[str, torch.Tensor], n_samples: int = 5):
    """Print basic information about the embeddings."""
    if not embeddings:
        print("No embeddings to inspect")
        return
    
    print(f"\n📊 Embedding Statistics:")
    print(f"   Number of sequences: {len(embeddings)}")
    
    # Get first embedding to check dimensions
    first_key = next(iter(embeddings))
    first_emb = embeddings[first_key]
    print(f"   Embedding dimension: {first_emb.shape[0]}")
    
    # Show sample IDs
    sample_ids = list(embeddings.keys())[:n_samples]
    print(f"   Sample IDs: {sample_ids}")
    
    # Basic statistics
    all_embeddings = torch.stack(list(embeddings.values()))
    print(f"   Mean embedding norm: {torch.norm(all_embeddings, dim=1).mean():.4f}")
    print(f"   Std embedding norm: {torch.norm(all_embeddings, dim=1).std():.4f}")

if __name__ == "__main__":
    # Example usage
    try:
        # Generate embeddings
        embeddings = generate_esm2_embeddings_from_csv(
            input_csv="merged_kcat_km_data.csv",
            output_file="esm_embeddings.pt",
            sequence_column="sequence",
            id_column=None,  # Will auto-generate IDs
            model_name="esm2_t33_650M_UR50D",  # 650M parameter model
            batch_size=16,
            filter_invalid=True,
            max_length=1000,  # Optional: filter very long sequences
            device=None  # Auto-detect
        )
        
        # Inspect the results
        inspect_embeddings(embeddings)
        
        # Example: Load embeddings later
        # loaded_embeddings = load_embeddings("esm_embeddings.pt")
        
    except Exception as e:
        print(f"❌ Error: {e}")