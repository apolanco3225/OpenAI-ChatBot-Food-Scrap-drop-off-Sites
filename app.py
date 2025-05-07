import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import openai
from openai import OpenAI
import gradio as gr
import tiktoken
from data_utils import load_data, find_similar_entries
from model import create_prompt, query_model, MAX_PROMPT_TOKENS

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Constants
EMBEDDINGS_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
MAX_PROMPT_TOKENS = 3000
MAX_ANSWER_TOKENS = 750

def generate_question_embeddings(input_question, dataframe):
    """Generate embeddings for the question and find similar entries"""
    input_data = dataframe.copy()
    
    embedding_question = client.embeddings.create(
        input=input_question,
        model=EMBEDDINGS_MODEL
    ).data[0].embedding

    embedding_question_array = np.array(embedding_question)

    input_data["distances"] = input_data["embeddings"].apply(
        lambda x: cosine(embedding_question_array, x)
    )

    input_data.sort_values(by="distances", ascending=True, inplace=True)

    return input_data

def process_query(question, df):
    """Process a user query and return the response"""
    # Find similar entries
    similar_entries = find_similar_entries(question, df)
    if similar_entries is None:
        return "Sorry, I encountered an error while processing your question."
    
    # Create prompt with context
    prompt = create_prompt(
        question=question,
        context_texts=similar_entries["text"].values,
        max_token_count=MAX_PROMPT_TOKENS
    )
    
    # Query the model
    return query_model(prompt)

def create_interface():
    """Create and return the Gradio interface"""
    # Load data
    df = load_data()
    if df is None:
        raise Exception("Failed to load data")
    
    # Create interface
    interface = gr.Interface(
        fn=lambda x: process_query(x, df),
        inputs=gr.Textbox(label="Ask a question about NYC Food Scrap Drop-off Sites"),
        outputs=gr.Textbox(label="Answer"),
        title="NYC Food Scrap Drop-off Sites Chatbot",
        description="Ask questions about food scrap drop-off locations in NYC",
        examples=[
            "When is Battery Park City Authority Rockefeller Park open?",
            "What is the address of Astoria Pug?",
            "What sites are open on weekends?",
        ]
    )
    
    return interface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch() 