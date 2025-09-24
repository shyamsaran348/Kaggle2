import os, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

model = SentenceTransformer("all-MiniLM-L6-v2")  # light & fast

train_emb = model.encode(train["body"].tolist(), batch_size=64, show_progress_bar=True)
test_emb  = model.encode(test["body"].tolist(),  batch_size=64, show_progress_bar=True)

np.save(os.path.join(DATA_DIR, "embeddings_train.npy"), train_emb)
np.save(os.path.join(DATA_DIR, "embeddings_test.npy"),  test_emb)
