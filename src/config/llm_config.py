from llama_index.llms.gemini import Gemini
from llama_index.core import Settings

def configure_language_model():
    """Configure the language model."""
    Settings.llm = Gemini(model="models/gemini-1.5-flash-latest", temperature=0)
