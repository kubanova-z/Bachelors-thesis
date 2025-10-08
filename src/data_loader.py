import pandas as pd

# data structure: Label, Text
def load_data(path: str):
    #load the cvs dataset
    #reads the cvs file into a dataframe
    df = pd.read_csv(path, names=["Category", "Text"], header=None)

    #shape of the file - number of rows and columns
    print("Dataset shape:", df.shape)
    

    #sample from each category
    print("\nCategories samples:")

    df_sample = df.groupby('Category', group_keys=False).head(1).sort_values(by='Category')


    for index, row in df_sample.iterrows():
        category = row['Category']
        # Extract the first 200 characters of the text
        description = row['Text'][:400]
        
        # Print the desired format: Category, newline, 200 chars of text
        print(f"\n{category}:")
        print(f"  {description}...")

    # reset column width to a standard value
    pd.set_option('display.max_colwidth', 50) 
    #representative from each category
   
    return df
