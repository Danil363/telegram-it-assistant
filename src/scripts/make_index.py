import faiss
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

f_name = 'data/rag_data.csv'

def make_faiss_index(f_name):
    df = pd.read_csv(f_name)
    texts = df['text']

    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_transformer.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  
    index.add(embeddings)
    print("Векторов в индексе:", index.ntotal)

    faiss.write_index(index, "data/tech_news_index.faiss")

if __name__ == '__main__':
    make_faiss_index(f_name)
    print('Creating index was sucecess')