import pandas as pd

# data structure: Label, Text
def load_data(path: str):
    #load the cvs dataset
    #reads the cvs file into a dataframe
    df = pd.read_csv(path)
    df["Text"] = df["title"].astype(str) + " " + df["content"].astype(str)

    df["Category"] = df["label"]

    # Keep only the two needed columns
    df = df[["Category", "Text"]]

    print("Dataset shape:", df.shape)
    

    #sample from each category
    print("\nCategories samples:")

    df_sample = df.groupby('Category', group_keys=False).head(1).sort_values(by='Category')


    for index, row in df_sample.iterrows():
        category = row['Category']
        #extract the first 200 characters of the text
        description = row['Text'][:400]
        

        print(f"\n{category}:")
        print(f"  {description}...")

    pd.set_option('display.max_colwidth', 50) 
   
    return df
