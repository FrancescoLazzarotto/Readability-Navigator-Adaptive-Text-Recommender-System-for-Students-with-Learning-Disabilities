import os
import pandas as pd
import numpy as np 
from src.user.model_user import user_model
from utils.io_utils import load_json, save_json
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

    def get_flesch(self, doc_id):
        idx = self.df.index[self.df["id"] == doc_id]
        if len(idx) == 0:
            raise ValueError("Document not found")
        return float(self.df.loc[idx[0], "flesch_score"])

    def gap_readability(self, user, flesch):
        target = user['target_readability']
        gap = abs(target - flesch)
        return gap, target, flesch

    def penalty(self, target, readability, alpha):
        if readability > target:
            return 1 + alpha
        return 1
        
    def theme_similarity(self, user, doc_id):
        topic_vector = np.array(user['topic_vector']).reshape(1, -1)
        _, emb = self.get_document(doc_id)
        emb = emb.reshape(1, -1)
        sim_score = cosine_similarity(topic_vector, emb)[0][0]
        return sim_score
        
    def recommender(self, user, doc_id):
        config = self.config
        eta = config['eta']
        zeta = config['zeta']
        alpha = config['alpha']
        
        flesch = self.get_flesch(doc_id)
        sim = self.theme_similarity(user, doc_id)
        gap, target, readability = self.gap_readability(user, flesch)
        
        penalty_score = self.penalty(target, readability, alpha)
        gap_penalized = gap * penalty_score
        
        score = eta * sim - zeta  * gap_penalized
        return score         


        

    

    
    
    
