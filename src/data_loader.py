import pandas as pd

# data structure: Label, Text
def load_data(path: str):
    df = pd.read_csv(path, names=["Category", "Text"], header=None)
    print("Dataset shape:", df.shape)
    print(df.head())
    return df
