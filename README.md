# NYC Food Scrap Drop-off Sites Chatbot

A chatbot that answers questions about food scrap drop-off locations in New York City using OpenAI's GPT-3.5 and embeddings models.

## Features

- Interactive web interface using Gradio
- Semantic search using OpenAI embeddings
- Context-aware responses using GPT-3.5
- Example questions and answers provided

## Prerequisites

- Python 3.9 or higher
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/OpenAI-ChatBot-Food-Scrap-drop-off-Sites.git
cd OpenAI-ChatBot-Food-Scrap-drop-off-Sites
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

1. Make sure you have the `embeddings.csv` file in the project directory (this contains the pre-computed embeddings for the food scrap drop-off sites data)

2. Run the application:
```bash
python app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860)

4. Start asking questions about NYC food scrap drop-off sites!

## Example Questions

- "When is Battery Park City Authority Rockefeller Park open?"
- "What is the address of Astoria Pug?"
- "What sites are open on weekends?"
- "Where can I find a drop-off site in Brooklyn?"
- "What items are not accepted at the sites?"

## Project Structure

- `app.py`: Main application file containing the chatbot logic and Gradio interface
- `embeddings.csv`: Pre-computed embeddings for the food scrap drop-off sites data
- `requirements.txt`: Python package dependencies
- `README.md`: Project documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.