import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load your catalog
df = pd.read_csv("shl_catalog.csv")

# Combine relevant fields into 1 string for each assessment
df['full_text'] = df.apply(lambda row: f"{row['Assessment Name']} - {row['Test Type']} - {row['Duration']} - Remote: {row['Remote Support']}, IRT: {row['IRT Support']}", axis=1)

# Load embedding model (small & fast)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["full_text"].tolist(), show_progress_bar=True)

# Save to disk
np.save("catalog_embeddings.npy", embeddings)
df.to_csv("shl_catalog_clean.csv", index=False)

# Save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, "faiss_index.index")
