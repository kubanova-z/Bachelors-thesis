import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def intra_category_similarity(X_vec, y_labels, class_to_idx):
    #X_vec = matrix of vectors

    #pd.DataFrame - data structure
    df_temp = pd.DataFrame(y_labels).rename(columns={y_labels.name: 'Category'})
    df_temp = df_temp.reset_index(drop=True)
    #drop = True -> each index starts from 0

    intra_similarity_score = {}

    print("\n" + "="*50)
    print("CATEGORY COSINE SIMILARITY")
    print("="*50)

    #loop for each category
    for category in class_to_idx.keys():
        #only rows belonging to current category
        index = df_temp[df_temp['Category'] == category].index

        #vectors for the category - sligned with index of X_vec
        category_vectors = X_vec[index]

        #check (are there enough samples)
        if category_vectors.shape[0] < 2:
            intra_similarity_score[category] = 0.0
            continue

        #pairwise similarity matrix
        #conversion of SciPy matrix into NumPy array
        #required to use the scikit learn function cosine_simila
        vectors = category_vectors.toarray()
        #calculate similarity
        similarity_matrix = cosine_similarity(vectors)

        #average of the score (upper triangle of the matrix, excluding the diagonal - all 1.0 document compared to itself)
        #upper triangle compared A to B, diagonal, compared A to A, lower triangle compared B to A - same as upper, not needed in the calculation
        upper_triangle_index = np.triu_indices(similarity_matrix.shape[0], k=1) #k=1 - diagonal


        #calculate the average
        if len(upper_triangle_index[0]) > 0:
            avg_score = similarity_matrix[upper_triangle_index].mean()
        else:
            avg_score = 0.0

        #assign the score to the correct category
        intra_similarity_score[category] = avg_score
        print(f"  {category:<25}: {avg_score:.4f}")

    return intra_similarity_score























""" def category_centroids(X_Vec, y_labels, class_to_idx):
    #mean TF-IDF vector for each category

    df_temp = pd.DataFrame(y_labels).rename(columns={y_labels.name: 'Category'})
    df_temp = df_temp.reset_index(drop=True)
    centroids = {}

    for category in class_to_idx.keys():
        index = df_temp[df_temp['Category'] == category].index
        category_vectors = X_Vec[index]
        centroid = category_vectors.mean(axis = 0)
        centroids[category] = centroid
    return centroids

def compare_category_similarity(centroids):
    #cosine similarity matrix for categories

    categories = list(centroids.keys())
    centroid_vectors = [centroids[cat].A.squeeze().reshape(1,-1) for cat in categories]
    
    X_matrix = np.vstack(centroid_vectors)
    similarity_matrix = cosine_similarity(X_matrix)

    df_similarity = pd.DataFrame(
        similarity_matrix,
        index=categories,
        columns=categories

    )

    print("\n" + "="*50)
    print("CATEGORY COSINE SIMILARITY MATRIX")
    print("="*50)
    print(df_similarity.round(4))
    
    return df_similarity """