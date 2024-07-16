from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings

def configure_embedding_model():
    """Configure the embedding model."""
    gemini_embedding_model = GeminiEmbedding(model_name="models/embedding-001")
    Settings.embed_model = gemini_embedding_model
