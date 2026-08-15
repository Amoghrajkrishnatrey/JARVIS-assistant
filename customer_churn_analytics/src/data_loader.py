import pandas as pd
import duckdb

def load_data(file_path: str) -> pd.DataFrame:
    print(f"Loading dataset from {file_path}...")
    return duckdb.query(f"SELECT * FROM '{file_path}'").df()

if __name__ == "__main__":
    print("Data Loader script ready.")
