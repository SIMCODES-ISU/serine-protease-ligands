#!/usr/bin/env python3
"""
Generate ESM2 embeddings from protein sequences in a CSV file.
Command-line interface for the ESM embedding generation utility.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the src directory to the Python path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))

try:
    from esm_embedding_utils import generate_esm2_embeddings_from_csv, load_embeddings, inspect_embeddings
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure esm_embedding_utils.py is in the src/ directory")
    sys.exit(1)

def validate_args(args):
    """Validate command line arguments."""
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Check output directory is writable
    output_dir = os.path.dirname(args.output) or "."
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"📁 Created output directory: {output_dir}")
        except Exception as e:
            print(f"❌ Error: Cannot create output directory '{output_dir}': {e}")
            sys.exit(1)
    
    if not os.access(output_dir, os.W_OK):
        print(f"❌ Error: Output directory '{output_dir}' is not writable")
        sys.exit(1)
    
    # Validate batch size
    if args.batch_size <= 0:
        print(f"❌ Error: Batch size must be positive, got {args.batch_size}")
        sys.exit(1)
    
    # Validate max length
    if args.max_length is not None and args.max_length <= 0:
        print(f"❌ Error: Max length must be positive, got {args.max_length}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM2 embeddings from protein sequences in a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate_embeddings.py --input data.csv --output embeddings.pt
  
  # Use specific model and batch size
  python generate_embeddings.py --input data.csv --output embeddings.pt \\
    --model esm2_t12_35M_UR50D --batch-size 32
  
  # Custom column names and filtering
  python generate_embeddings.py --input data.csv --output embeddings.pt \\
    --sequence-column protein_seq --id-column protein_id \\
    --max-length 500 --no-filter-invalid
  
  # Inspect existing embeddings
  python generate_embeddings.py --inspect embeddings.pt

Available ESM Models:
  esm2_t6_8M_UR50D      - 8M parameters (fastest)
  esm2_t12_35M_UR50D    - 35M parameters  
  esm2_t30_150M_UR50D   - 150M parameters
  esm2_t33_650M_UR50D   - 650M parameters (default)
  esm2_t36_3B_UR50D     - 3B parameters (most accurate)
        """
    )
    
    # Main arguments
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        help="Path to input CSV file containing protein sequences"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        help="Path to save the output embeddings (.pt file)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model", "-m",
        type=str, 
        default="esm2_t33_650M_UR50D",
        help="ESM model name (default: esm2_t33_650M_UR50D)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for computation (default: auto)"
    )
    
    # Data processing options
    parser.add_argument(
        "--sequence-column",
        type=str,
        default="sequence",
        help="Name of the column containing protein sequences (default: sequence)"
    )
    
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Name of the column containing sequence IDs (optional)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for processing sequences (default: 16)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum sequence length to process (sequences longer than this will be filtered out)"
    )
    
    # Filtering options
    parser.add_argument(
        "--no-filter-invalid",
        action="store_true",
        help="Don't filter sequences with invalid amino acids"
    )
    
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Don't remove duplicate sequences"
    )
    
    # Utility options
    parser.add_argument(
        "--inspect",
        type=str,
        help="Inspect existing embeddings file instead of generating new ones"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it exists"
    )
    
    args = parser.parse_args()
    
    # Handle inspection mode
    if args.inspect:
        if not os.path.exists(args.inspect):
            print(f"❌ Error: File '{args.inspect}' not found")
            sys.exit(1)
        
        try:
            embeddings = load_embeddings(args.inspect)
            inspect_embeddings(embeddings, n_samples=10)
            return
        except Exception as e:
            print(f"❌ Error inspecting embeddings: {e}")
            sys.exit(1)
    
    # Require input and output for generation mode
    if not args.input or not args.output:
        parser.error("--input and --output are required (unless using --inspect)")
    
    # Validate arguments
    validate_args(args)
    
    # Check if output file exists
    if os.path.exists(args.output) and not args.force:
        response = input(f"⚠️  Output file '{args.output}' already exists. Overwrite? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Operation cancelled.")
            sys.exit(0)
    
    # Set device
    device = None if args.device == "auto" else args.device
    
    # Print configuration if verbose
    if args.verbose:
        print("🔧 Configuration:")
        print(f"   Input file: {args.input}")
        print(f"   Output file: {args.output}")
        print(f"   Model: {args.model}")
        print(f"   Device: {args.device}")
        print(f"   Sequence column: {args.sequence_column}")
        print(f"   ID column: {args.id_column or 'auto-generated'}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Max length: {args.max_length or 'no limit'}")
        print(f"   Filter invalid: {not args.no_filter_invalid}")
        print(f"   Deduplicate: {not args.no_deduplicate}")
        print()
    
    try:
        # Generate embeddings
        embeddings = generate_esm2_embeddings_from_csv(
            input_csv=args.input,
            output_file=args.output,
            sequence_column=args.sequence_column,
            id_column=args.id_column,
            model_name=args.model,
            batch_size=args.batch_size,
            filter_invalid=not args.no_filter_invalid,
            max_length=args.max_length,
            device=device
        )
        
        # Show summary
        if args.verbose:
            print("\n📊 Generation Summary:")
            inspect_embeddings(embeddings, n_samples=5)
        
        print(f"\n🎉 Successfully generated embeddings!")
        print(f"   Sequences processed: {len(embeddings)}")
        print(f"   Output saved to: {args.output}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error generating embeddings: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()