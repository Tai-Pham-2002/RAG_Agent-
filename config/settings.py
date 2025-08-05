# config/settings.py
import os
from dotenv import load_dotenv
load_dotenv()

# API Keys (replace with actual keys or use environment variables)
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

# Model settings
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen/qwen3-32b"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 0
RETRIEVER_K = 2

# Data URLs
DATA_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]