import os
import tiktoken
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Constants
EMBEDDINGS_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
MAX_PROMPT_TOKENS = 3000
MAX_ANSWER_TOKENS = 750

def generate_embeddings(input_text):
    """Generate embeddings for input text using OpenAI's API"""
    try:
        response = client.embeddings.create(
            input=input_text,
            model=EMBEDDINGS_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def create_prompt(question, context_texts, max_token_count):
    """Create a prompt for the chat model"""
    tokenizer = tiktoken.get_encoding("cl100k_base")
    
    prompt_template = """
Answer the question based on the context below, and if the question
can't be answered based on the context, say "I don't know"

Context: 

{}

---

Question: {}
Answer:"""
    
    num_tokens_template = len(tokenizer.encode(prompt_template))
    num_tokens_question = len(tokenizer.encode(question))
    current_token_count = num_tokens_template + num_tokens_question
    
    context = []
    for text in context_texts:
        text_token_count = len(tokenizer.encode(text))
        current_token_count += text_token_count
        
        if current_token_count <= max_token_count:
            context.append(text)
        else:
            break

    return prompt_template.format("\n\n###\n\n".join(context), question)

def query_model(prompt):
    """Query the OpenAI chat model with the given prompt"""
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=MAX_ANSWER_TOKENS
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error querying model: {e}")
        return "Sorry, I encountered an error while processing your question." 