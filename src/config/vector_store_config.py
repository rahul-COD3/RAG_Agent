import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_parse import LlamaParse
import nest_asyncio

def initialize_vector_store():
    """Configure and initialize the ChromaDB for vector storage."""
    nest_asyncio.apply()
    documents = LlamaParse(result_type="markdown").load_data("./src/data/knowledge.pdf")
    db = chromadb.PersistentClient(path="./src/chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index.as_query_engine()
