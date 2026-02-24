import zipfile
import pandas as pd
import io

zip_path = "/Users/drs/Projects/DTI/Backend/app/data/raw/BindingDB_All_202511_tsv.zip"

with zipfile.ZipFile(zip_path, 'r') as z:
    # find the TSV inside the zip
    tsv_name = [n for n in z.namelist() if n.endswith(".tsv")][0]

    with z.open(tsv_name) as f:
        df_names = pd.read_csv(
            io.TextIOWrapper(f, encoding="utf-8"),
            sep="\t",
            usecols=["BindingDB Ligand Name"],
            nrows=100_000
        )

print("Non-null count:",
      df_names["BindingDB Ligand Name"].notna().sum())

print("Unique non-null:",
      df_names["BindingDB Ligand Name"].nunique())