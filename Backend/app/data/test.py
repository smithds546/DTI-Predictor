import zipfile
import pandas as pd
import io

# Path to your zip
zip_path = "/Users/drs/Projects/DTI/Backend/app/data/raw/BindingDB_All_202511_tsv.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    # Find the TSV inside the zip
    tsv_name = [name for name in z.namelist() if name.endswith('.tsv')][0]
    with z.open(tsv_name) as f:
        # Read first few rows only
        df_preview = pd.read_csv(io.TextIOWrapper(f, encoding='utf-8'), sep='\t', nrows=5)
        print("Columns in TSV:", df_preview.columns.tolist())
        print(df_preview.head())
