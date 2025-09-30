import pandas as pd

# data structure: Label, Text
def load_data(path: str):
    #load the cvs dataset
    #reads the cvs file into a dataframe
    df = pd.read_csv(path, names=["Category", "Text"], header=None)

    #shape of the file - number of rows and columns
    print("Dataset shape:", df.shape)

    #first 5 rows
    print(df.head())
    return df
