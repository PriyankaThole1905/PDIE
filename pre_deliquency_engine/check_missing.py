import pandas as pd
import glob
import os

data_dir = r"E:\Microsoft VS Code\PDIE_new\pre_deliquency_engine\pdie_feature_store"
files = glob.glob(os.path.join(data_dir, "*.parquet"))

print(f"Checking {len(files)} files in {data_dir}\n")

for f in files:
    try:
        df = pd.read_parquet(f)
        print(f"--- {os.path.basename(f)} ---")
        print(f"Rows: {len(df)}")
        print("Missing values per column:")
        print(df.isna().sum())
        print("\n")
    except Exception as e:
        print(f"Error reading {f}: {e}")
