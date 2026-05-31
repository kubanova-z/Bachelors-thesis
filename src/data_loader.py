import pandas as pd

# data structure: Label, Text
def load_data(path: str):
    # load the cvs dataset
    # reads the cvs file into a dataframe
    df = pd.read_csv(path, names=["Category", "Text"], header=None)

    #shape of the file - number of rows and columns
    print("Dataset shape:", df.shape)

    # sample from each category
    print("\nCategories samples:")

    df_sample = df.groupby('Category', group_keys=False).head(1).sort_values(by='Category')

    for index, row in df_sample.iterrows():
        category = row['Category']
        # sample from text
        description = row['Text'][:400]
        
        print(f"\n{category}:")
        print(f"  {description}...")

    pd.set_option('display.max_colwidth', 50) 

   
    return df
