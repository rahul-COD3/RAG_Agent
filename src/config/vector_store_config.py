import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex


def initialize_vector_store(collection_name: str, documents = None):
    """Configure and initialize the ChromaDB for vector storage."""
    db = chromadb.PersistentClient(path="./src/chroma_db")
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
  # Conditionally load documents into the vector store
    if documents is None:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    else:
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

    return index.as_query_engine()
