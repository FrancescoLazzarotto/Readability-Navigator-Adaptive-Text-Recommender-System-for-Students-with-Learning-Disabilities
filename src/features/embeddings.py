import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from utils.io_utils import load_data, save_pickle, load_pickle


path = r"C:\Users\checc\OneDrive\Desktop\readability-navigator\data\processed\onestop_nltk_features.csv"
PATH_OUTPUT_DIR = "data_artifacts/"
try: 
    dataframe = load_data(path)
except FileNotFoundError:
    raise FileNotFoundError(f"File non trovato nel path {path}")

sentences = dataframe['testo']


def model_embedding():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model 

model = model_embedding()

def sentences_embedding(sentences, model):
    
    embedding = model.encode(
        sentences,
        truncate_dim = 512
                            )
    return embedding

def topic_embedding(topic_text, model):
    embedding = model.encode([topic_text], truncate_dim=512)
    return embedding[0].tolist()


def similarity_embedding(embedding, model):     
    similarity = model.similarity(embedding, embedding)
    return similarity

def clustering(df, embedding, k = 10, random_state = 42):
    kmeans = KMeans(n_clusters=k, random_state = random_state)
    labels = kmeans.fit_predict(embedding)
    df["topic_cluster"] = labels
    genre_descriptions = cluster_label(df, k)
    save_pickle(f'{PATH_OUTPUT_DIR}kmeans_model.pickle', kmeans)
    save_pickle(f'{PATH_OUTPUT_DIR}genre_descriptions.pickle', genre_descriptions)
    return df, kmeans


def cluster_label(df, k, n_keywords=5):
    
    docs_per_cluster = df.groupby(['topic_cluster'], as_index = False)
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    c_tf_idf = tfidf.fit_transform(docs_per_cluster['testo'])
    feature_names = tfidf.get_feature_names_out()
    
    cluster_keywords = {}
    
    for i in range(k):
        cluster_vector = c_tf_idf[i]
        sorted_indices = cluster_vector.toarray().flatten().argsort()[::-1]
        top_words = [feature_names[idx] for idx in sorted_indices[:n_keywords]]
        cluster_keywords[i] = top_words
        
    return cluster_keywords

def user_seg(user_topic_vector, kmeans, genre_descriptions):
    
    user_vec = np.array(user_topic_vector).reshape(1, -1)
    cluster_id = kmeans.predict(user_vec)[0]
    return genre_descriptions[cluster_id]    

#embedding = sentences_embedding(sentences, model)
#save_pickle('doc_embedding.pickle', embedding)




