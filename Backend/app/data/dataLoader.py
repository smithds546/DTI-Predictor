"""
BindingDB dataset loader for DTI prediction
Loads and processes the FULL BindingDB dataset from local files.
This method is robust, reproducible, and memory-efficient.
"""

import pandas as pd
import os
import re
from pathlib import Path
import zipfile
import io

# --- Parameters ---
BINDER_THRESHOLD_NM = 800.0
NONBINDER_THRESHOLD_NM = 5000.0
CHUNK_SIZE = 100000              # Process 100k rows at a time

class BindingDBLoader:
    """Load and process BindingDB from local downloaded files"""

    def __init__(self, data_dir: str = "Backend/app/data"):
        # We'll read from 'raw' and write to 'loaded'
        self.raw_dir = Path(data_dir) / "raw"
        self.output_dir = Path(data_dir) / "loaded" # Using your 'loaded' folder for output
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save_processed_dataset(self, df: pd.DataFrame, filename: str = 'bindingdb_processed.csv'):
        """Save loaded dataset to CSV, appending a timestamp to the filename."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        if filename:
            base, ext = os.path.splitext(filename)
            filename_with_timestamp = f"{base}_{timestamp}{ext}"
        else:
            filename_with_timestamp = f"bindingdb_processed_{timestamp}.csv"

        filepath = self.output_dir / filename_with_timestamp
        df.to_csv(filepath, index=False)
        print(f"\nSaved loaded dataset to {filepath}")
        return filepath

    def parse_fasta(self, fasta_path: Path) -> dict:
        """
        Parses a FASTA file into a dictionary.
        Keys: target names (e.g., "Thymidine kinase")
        Values: amino acid sequences
        """
        print(f"Loading protein sequences from {fasta_path}...")
        sequences = {}
        current_id = None
        current_seq = []

        try:
            with open(fasta_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        # Save previous sequence
                        if current_id and current_seq:
                            sequences[current_id] = "".join(current_seq)
                            current_seq = []

                        # Parse new header
                        header = line[1:].strip()
                        header = re.sub(r"mol:protein length:\d+\s*", "", header)
                        protein_name = re.sub(r"^p\d+\s*", "", header).strip()
                        current_id = protein_name
                    elif current_id:
                        current_seq.append(line)

                # Save the last sequence
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)

        except FileNotFoundError:
            print(f"Error: FASTA file not found at {fasta_path}")
            return {}
        except Exception as e:
            print(f"Error parsing FASTA file: {e}")
            return {}

        print(f"Successfully loaded {len(sequences)} sequences.")
        return sequences

    def clean_affinity_value(self, value_str):
        """Cleans affinity strings like '>', '<', or '='."""
        if pd.isna(value_str):
            return None
        # Remove common prefixes
        value_str = str(value_str).strip().replace(">", "").replace("<", "").replace("=", "").replace("~", "")
        try:
            return float(value_str)
        except ValueError:
            return None

    def build_dataset_from_files(self, tsv_zip_name: str, fasta_name: str):
        """
        Main function to build the DTI dataset from local files.
        Processes the large TSV in chunks to save memory.
        """
        fasta_path = self.raw_dir / fasta_name
        tsv_zip_path = self.raw_dir / tsv_zip_name

        # 1. Load Sequences (Small file, load all at once)
        seq_dict = self.parse_fasta(fasta_path)
        if not seq_dict:
            print("Could not load sequences, aborting.")
            return

        # 2. Process Affinity Data (Large file, process in chunks)
        print(f"Loading affinity data from {tsv_zip_path} in chunks...")
        processed_chunks = []

        try:
            with zipfile.ZipFile(tsv_zip_path, 'r') as z:
                # Find the actual .tsv file inside the zip
                tsv_filename = [name for name in z.namelist() if name.endswith('.tsv')][0]
                print(f"Found {tsv_filename} inside zip.")

                with z.open(tsv_filename) as f:
                    # Wrap in io.TextIOWrapper for pandas
                    f_text = io.TextIOWrapper(f, encoding='utf-8')

                    for i, chunk in enumerate(pd.read_csv(f_text, sep='\t', on_bad_lines='skip', low_memory=False, chunksize=CHUNK_SIZE)):
                        print(f"  ... processing chunk {i+1} (approx {i*CHUNK_SIZE/1_000_000:.1f}M rows)")

                        # Filter for Ki or Kd
                        df_chunk = chunk[chunk['Ki (nM)'].notna() | chunk['Kd (nM)'].notna()].copy()
                        if df_chunk.empty:
                            continue

                        df_chunk['Affinity_Value_nM_Str'] = df_chunk['Ki (nM)'].fillna(df_chunk['Kd (nM)'])

                        # --- Identify the correct UniProt/Target ID column dynamically ---
                        possible_cols = [
                            'Target ChEMBL ID derived from UniProt',
                            'UniProt (Primary ID of Target)',
                            'Target UniProt ID',
                            'UniProt ID',
                            'Target Name Assigned by Curator or DataSource',
                            'Target Name',  # <-- NEW fallback
                        ]
                        protein_col = next((c for c in possible_cols if c in df_chunk.columns), None)
                        if not protein_col:
                            raise KeyError(
                                f"No UniProt or Target Name column found in BindingDB TSV. Available columns: {list(df_chunk.columns)[:25]}"
                            )

                        # Keep only necessary columns
                        df_chunk = df_chunk[['Ligand SMILES', protein_col, 'Affinity_Value_nM_Str']]
                        df_chunk.columns = ['drug_smiles', 'protein_uniprot', 'Affinity_Str']
                        df_chunk.dropna(subset=['drug_smiles', 'protein_uniprot', 'Affinity_Str'], inplace=True)

                        # Clean Affinity Values
                        df_chunk['affinity_value_nm'] = df_chunk['Affinity_Str'].apply(self.clean_affinity_value)
                        df_chunk.dropna(subset=['affinity_value_nm'], inplace=True)

                        # Create Binary Labels
                        # -1 = invalid, 0 = non-binder, 1 = binder
                        df_chunk['interaction'] = -1
                        df_chunk.loc[df_chunk['affinity_value_nm'] < BINDER_THRESHOLD_NM, 'interaction'] = 1
                        df_chunk.loc[df_chunk['affinity_value_nm'] > NONBINDER_THRESHOLD_NM, 'interaction'] = 0

                        # Keep only the rows that meet our criteria (0 or 1)
                        df_chunk = df_chunk[df_chunk['interaction'].isin([0, 1])]

                        if not df_chunk.empty:
                            processed_chunks.append(df_chunk)

        except FileNotFoundError:
            print(f"Error: TSV Zip file not found at {tsv_zip_path}")
            print(f"Please make sure you downloaded '{tsv_zip_name}' and placed it in '{self.raw_dir}'")
            return
        except Exception as e:
            print(f"Error processing TSV file: {e}")
            return

        if not processed_chunks:
            print("No valid Ki/Kd data found in the file matching your criteria.")
            return

        # 3. Combine Chunks
        print("Combining loaded chunks...")
        df_final = pd.concat(processed_chunks)
        print(f"Total loaded interactions: {len(df_final)}")

        print("\n--- Debug: FASTA keys preview ---")
        print(list(seq_dict.keys())[:10])

        print("\n--- Debug: Target names in dataset ---")
        print(df_final['protein_uniprot'].dropna().unique()[:10])

        # 4. Map Sequences
        print("Mapping protein sequences...")

        # Try mapping by name (or UniProt if present)
        if 'protein_uniprot' in df_final.columns:
            df_final['protein_sequence'] = df_final['protein_uniprot'].map(seq_dict)
        elif 'Target Name' in df_final.columns:
            df_final['protein_sequence'] = df_final['Target Name'].map(seq_dict)
        else:
            print("Warning: Could not find column to map sequences.")
            df_final['protein_sequence'] = None

        df_final.dropna(subset=['protein_sequence'], inplace=True)
        print(f"Interactions with valid sequences: {len(df_final)}")

        # 5. Final Cleanup
        df_final = df_final[['drug_smiles', 'protein_sequence', 'protein_uniprot', 'affinity_value_nm', 'interaction']].drop_duplicates()
        df_final.reset_index(drop=True, inplace=True)

        print("\n--- Dataset Build Complete ---")
        explore_dataset(df_final)

        # 6. Save final file
        self.save_processed_dataset(df_final, filename="bindingdb_offline_processed.csv")


def explore_dataset(df: pd.DataFrame):
    """Explore dataset statistics"""
    print("\n" + "=" * 60)
    print("DATASET EXPLORATION")
    print("=" * 60)

    if df is None or df.empty:
        print("\nNo data available to explore. DataFrame is empty.")
        return

    print("\n--- Basic Statistics ---")
    print(f"Total samples: {len(df)}")
    if 'interaction' in df.columns:
        print(f"Positive interactions (1): {df['interaction'].sum()} ({df['interaction'].mean() * 100:.2f}%)")
        print(f"Negative interactions (0): {(1 - df['interaction']).sum()} ({(1 - df['interaction'].mean()) * 100:.2f}%)")

    print("\n--- Drug Statistics ---")
    if 'ligand_smiles' in df.columns:
        print(f"Unique drugs: {df['ligand_smiles'].nunique()}")

    print("\n--- Protein Statistics ---")
    if 'protein_uniprot' in df.columns:
        print(f"Unique proteins: {df['protein_uniprot'].nunique()}")

    print("\n--- Affinity Statistics (nM) ---")
    if 'affinity_value_nm' in df.columns:
        print(df['affinity_value_nm'].describe())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    return


# Main execution script
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Manually change these filenames to match what you downloaded
    # Find the .zip file in your 'raw' folder and put its name here
    TSV_ZIP_FILENAME = "BindingDB_All_202511_tsv.zip"  # e.g., "BindingDB_All_202511_tsv.zip"
    FASTA_FILENAME = "BindingDBTargetSequences.fasta"
    # --- IMPORTANT ---

    print("\n" + "=" * 60)
    print("Loading BindingDB data from LOCAL FILES (Ki and Kd)")
    print("This is the robust, offline method.")
    print("=" * 60)

    # Assumes your 'data' folder is 'Backend/app/data'
    loader = BindingDBLoader(data_dir="/Users/drs/Projects/DTI/Backend/app/data")

    # Check if files exist before running
    raw_path = Path("/Users/drs/Projects/DTI/Backend/app/data/raw")
    if not (raw_path / TSV_ZIP_FILENAME).exists():
        print(f"FATAL ERROR: File not found: {raw_path / TSV_ZIP_FILENAME}")
        print(f"Please check the filename and update the TSV_ZIP_FILENAME variable in this script (line 228).")
    elif not (raw_path / FASTA_FILENAME).exists():
        print(f"FATAL ERROR: File not found: {raw_path / FASTA_FILENAME}")
        print(f"Please download 'BindingDBTargetSequences.fasta' and place it in '{raw_path}'.")
    else:
        loader.build_dataset_from_files(TSV_ZIP_FILENAME, FASTA_FILENAME)

        print("\n" + "=" * 60)
        print("✓ Local dataset loaded successfully!")
        print(f"✓ Find your new, filtered file in: Backend/app/data/loaded/")
        print("=" * 60)


