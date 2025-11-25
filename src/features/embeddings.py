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



#embedding = sentences_embedding(sentences, model)
#save_pickle('doc_embedding.pickle', embedding)




