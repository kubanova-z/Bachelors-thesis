import pandas as pd

expected_columns = ["title", "content", "label"]
csv_path = "/home/xkubanova_126831/bakalarka/nbs_binary/data/ekosentiment_titles_binary.csv"

try:
    # Try reading the CSV normally
    df = pd.read_csv(csv_path)
    
    print("\nFile loaded successfully!")
    print(f"Detected columns: {list(df.columns)}")
    
    # Check if column count matches expectations
    if list(df.columns) == expected_columns:
        print("\n✅ CSV format is correct.")
    else:
        print("\n⚠️ WARNING: CSV does not match the expected structure.")
        print(f"Expected columns: {expected_columns}")
        print(f"Found columns:    {list(df.columns)}")
        
        # Additional diagnostics
        print("\nPossible issues:")
        print("- Commas inside text fields without quotes")
        print("- Wrong delimiter (maybe ';' instead of ',')?")
        print("- Broken rows")
        
        # Try reading with semicolon delimiter in case the file uses it
        try:
            df_alt = pd.read_csv(csv_path, delimiter=';')
            print("\nTried loading with ';' delimiter:")
            print(f"Columns: {list(df_alt.columns)}")
        except Exception:
            pass

except Exception as e:
    print("\n❌ Failed to load CSV:")
    print(str(e))
