import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
import openai
import streamlit as st
import os

openai.api_key = st.secrets["openai_api_key"]

def load_data():
    df = pd.read_csv("shl_catalog_clean.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    if os.path.exists("catalog_embeddings.npy"):
        embeddings = np.load("catalog_embeddings.npy")
    else:
        embeddings = model.encode(df['Description'].tolist(), show_progress_bar=True)
        np.save("catalog_embeddings.npy", embeddings)

    index = NearestNeighbors(n_neighbors=5, metric='cosine')
    index.fit(embeddings)

    return df, embeddings, index

def get_top_k(query, k=5):
    df, embeddings, index = load_data()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = model.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding).reshape(1, -1)  # Ensure correct shape
    distances, indices = index.kneighbors(query_embedding, return_distance=True)
    top_k_df = df.iloc[indices[0]]
    return top_k_df


def generate_response(context, query):
    prompt = f"""You are an expert assessment recommendation system. Based on the context below, respond to the user's query with the most relevant assessment(s).

Context:
{context}

Query:
{query}

Response:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You provide assessment recommendations based on SHL catalog data."},
            {"role": "user", "content": prompt}
        ]
    )

    return response['choices'][0]['message']['content']
