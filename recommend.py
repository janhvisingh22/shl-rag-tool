import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai

openai.api_key = "your-openai-key"

def load_data():
    df = pd.read_csv("shl_catalog_clean.csv")
    embeddings = np.load("catalog_embeddings.npy")
    index = faiss.read_index("faiss_index.index")
    return df, embeddings, index

def get_top_k(query, k=5):
    df, _, index = load_data()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return df.iloc[I[0]]

def generate_response(query, top_df):
    context = "\n".join(top_df['full_text'].tolist())
    prompt = f"You are a helpful assistant. Based on this query: '{query}', choose the most relevant assessments:\n\n{context}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
