"""Configuration constants for Content Radar."""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Embedding
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

# Buffer & Clustering
BUFFER_SIZE = 100
N_CLUSTERS = 5
CLUSTER_REFRESH_INTERVAL = 2.0  # seconds

# Anomaly Detection
GROWTH_THRESHOLD = 8      # comments per window
DENSITY_THRESHOLD = 0.80  # cosine similarity

# Stream
NORMAL_COMMENTS_BEFORE_RAID = 50
RAID_SIZE = 25
STREAM_DELAY = 1  # seconds between comments

# Groq
GROQ_MODEL = "llama-3.1-8b-instant"  # for rule generation
SAFEGUARD_MODEL = "openai/gpt-oss-safeguard-20b"  # gpt-oss-safeguard equivalent on Groq

# SP Level Colors (for visualization)
SP_COLORS = {
    "SP0": "#22C55E",  # green - safe
    "SP1": "#84CC16",  # lime - borderline
    "SP2": "#EAB308",  # yellow - likely spam
    "SP3": "#F97316",  # orange - spam
    "SP4": "#EF4444",  # red - severe
    "pending": "#9CA3AF"  # gray - not yet classified
}

# Binary classification colors
BINARY_COLORS = {
    "Safe": "#22C55E",     # green
    "Unsafe": "#EF4444",   # red
    "pending": "#9CA3AF"   # gray
}

# SP levels that count as "Unsafe"
UNSAFE_SP_LEVELS = {"SP2", "SP3", "SP4"}
