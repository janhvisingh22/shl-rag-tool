from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import pickle

# Load the model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_data():
    df = pd.read_csv("shl_catalog_clean.csv")
    embeddings = np.load("catalog_embeddings.npy")
    with open("nn_index.pkl", "rb") as f:
        index = pickle.load(f)
    return df, embeddings, index

def get_top_k(query, k=5):
    df, _, index = load_data()
    query_embedding = model.encode(query)
    query_embedding = query_embedding.reshape(1, -1)  # reshape to match kneighbors input
    distances, indices = index.kneighbors(query_embedding, return_distance=True)
    top_k_df = df.iloc[indices[0]].copy()
    top_k_df["similarity"] = 1 - distances[0]
    return top_k_df
