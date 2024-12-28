# Wisdom Bot 🤖

A chatbot that answers questions using insights from Marcus Aurelius's *Meditations* and the *Tao Te Ching*. It uses RAG (Retrieval Augmented Generation) to find relevant passages from these texts and generate responses based on their wisdom.

## How It Works 🔍

The bot processes each question in three steps:

1. Reformulates the user's question to better match the philosophical texts using Gemini 1.5 Flash
2. Searches a vector database (Qdrant) to find relevant passages from both texts (you'll need to embed these texts to your own Vector datbase to reproduce this)
3. Generates a summary response using GPT-4o Mini, incorporating the found passages as context to guide the summary
4. Includes the top matching quote from Meditations and Tao Te Ching as part of the answer to demonstrate that sources were pulled to generate a resopnse and the answer is not just from a call to an LLM.

## Setup 🚀

### Requirements

- Python 3.11+
- OpenAI API key
- Google API key (for Gemini)
- Qdrant instance (cloud or local)

### Quick Start

1. Clone and install:
```bash
git clone https://github.com/yourusername/wisdom-bot.git
cd wisdom-bot
pip install -r requirements.txt
```

2. Create a `.streamlit/secrets.toml` file if deploying locally (Streamlit prefers this to using a .env file):
```toml
OPENAI_API_KEY = "your_openai_api_key"
GOOGLE_API_KEY = "your_google_api_key"
QDRANT_URL = "your_qdrant_url"
QDRANT_API_KEY = "your_qdrant_api_key"
```

!!!Note: When deploying to Streamlit Cloud, add these same values to your app's secrets management in the Streamlit Cloud dashboard instead of using a local secrets.toml file.

3. Run:
```bash
streamlit run app.py
```

## Technical Details 🛠️

The application uses:

- Streamlit for the front-end web interface
- LlamaIndex for RAG implementation
- GPT-4o Mini for generating responses
- Gemini 1.5 Flash for query processing
- Qdrant as the vector database
- Gemini text-embedding-004 for embedding the original texts as well as user queries


## License 📄

This project is under the MIT License. See [LICENSE](LICENSE) for details.

---

Built with Python, LlamaIndex, and ancient wisdom ✨