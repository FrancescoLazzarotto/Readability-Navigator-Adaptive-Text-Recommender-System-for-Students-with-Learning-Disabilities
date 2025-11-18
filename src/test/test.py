import pandas as pd
import numpy as np
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.recommender.recommender_engine import RecommenderEngine


df = pd.DataFrame({
    "id": [1, 2, 3],
    "testo": ["ciao come stai", "oggi piove", "machine learning"],
    "flesch_score": [70, 55, 40],
    "topic_cluster": [0, 1, 0]
})


embedding = np.random.rand(3, 384)

config = {
    "eta": 1.0,
    "zeta": 0.8,
    "alpha": 0.4
}

user = {
    "user_id": 10,
    "target_readability": 60,
    "topic_vector": list(np.random.rand(384)),
    "history": []
}


engine = RecommenderEngine(
    df=df,
    embedding=embedding,
    config=config,
    user_id=10,
    profile_path="user10.json"
)


doc_id = 1
score = engine.recommender(user, doc_id)

print("Score:", score)
