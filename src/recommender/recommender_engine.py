import os
import pandas as pd
import numpy as np 
from user.model_user import user_model
from utils.io_utils import load_yamal, load_json, save_json, load_pickle
from sklearn.metrics.pairwise import cosine_similarity


class RecommenderEngine():
    def __init__(self, df, embedding, config, user_id, profile_path):
        self.df = df
        self.embedding = embedding
        self.config = config
        self.user_id = user_id
        self.profile_path = profile_path

    def profile(self):
        if os.path.exists(self.profile_path):
            return load_json(self.profile_path)
        chosen_cluster = self.df["topic_cluster"].unique().tolist()[:3]
        profile = user_model(self.user_id, chosen_cluster, self.df, self.embedding)
        save_json(profile, self.profile_path)
        return profile

    def catalog(self, profile):
        tol = self.config["tol"]
        target = profile["target_readability"]
        history = set(profile["history"])
        df = self.df[~self.df["id"].isin(history)]
        df = df[np.abs(df["flesch_score"] - target) <= tol]
        return df
    
    def get_document(self, doc_id):
        idx = self.df.index[self.df["id"] == doc_id][0]
        testo = self.df.loc[idx, "testo"]
        emb = self.embedding[idx]
        return testo, emb

    def get_fusch(self):
        return NotImplementedError
    
    
    def theme_similarity(self, user, doc_id):
        topic_vector = np.array(user['topic_vector']).reshape(1, -1)
        _, emb = self.get_document(doc_id)
        emb = emb.reshape(1, -1)
        sim_score = cosine_similarity(topic_vector, emb)[0][0]
        return sim_score
        
