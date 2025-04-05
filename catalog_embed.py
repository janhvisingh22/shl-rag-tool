import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pickle

# Load your catalog CSV
df = pd.read_csv("shl_catalog_clean.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode text (assuming relevant column is 'text' or 'description')
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
np.save("catalog_embeddings.npy", embeddings)

# Create and save nearest neighbors index
index = NearestNeighbors(n_neighbors=5, metric="cosine")
index.fit(embeddings)
with open("nn_index.pkl", "wb") as f:
    pickle.dump(index, f)
