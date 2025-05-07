import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from model import generate_embeddings

def load_data():
    """Load and prepare the embeddings data"""
    try:
        data = pd.read_csv("embeddings.csv", index_col=0)
        data["embeddings"] = data["embeddings"].apply(eval).apply(np.array)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def find_similar_entries(input_question, dataframe):
    """Find similar entries in the dataframe based on question embeddings"""
    input_data = dataframe.copy()
    
    # Generate embeddings for the question
    embedding_question = generate_embeddings(input_question)
    if embedding_question is None:
        return None
    
    embedding_question_array = np.array(embedding_question)

    # Calculate distances and sort
    input_data["distances"] = input_data["embeddings"].apply(
        lambda x: cosine(embedding_question_array, x)
    )

    input_data.sort_values(by="distances", ascending=True, inplace=True)

    return input_data 