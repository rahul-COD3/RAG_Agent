from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from src.config.vector_store_config import initialize_vector_store
from src.config.llm_config import configure_language_model
from src.config.embedding_model_config import configure_embedding_model
from llama_index.core import Settings
from src.config.data_reader import load_local_data, load_web_data
from src.config.vector_store_config import initialize_vector_store
from src.consts.disc_consts import HOTEL_TOOL_DESCRIPTION
import chromadb

def check_hotel_reservation_availability(date: str) -> str:
    """Check if hotel reservations are available for the given date."""
    if date.lower() == "tomorrow":
        return "Yes, reservations are available for tomorrow."
    else:
        return "Sorry, reservations are only available for tomorrow at the moment."

def hotel_query_engine():
    if isCollectionExists("hotel_knowledge"):
        return initialize_vector_store("hotel_knowledge")
    else:
        hotel_data = load_local_data()
        return initialize_vector_store("hotel_knowledge", hotel_data)

def gujarat_query_engine():
    if isCollectionExists("gujarat_knowledge"):
        return initialize_vector_store("gujarat_knowledge")
    else:
        gujarat_data = load_web_data("https://www.holidify.com/state/gujarat/top-destinations-places-to-visit.html")
        return initialize_vector_store("gujarat_knowledge", gujarat_data)

def bihar_query_engine():
    if isCollectionExists("bihar_knowledge"):
        return initialize_vector_store("bihar_knowledge")
    else:
        bihar_data = load_web_data("https://www.holidify.com/state/bihar/top-destinations-places-to-visit.html")
        return initialize_vector_store("bihar_knowledge", bihar_data)

def initialize_agent():
    """Define and initialize the ReActAgent with necessary tools."""
    configure_language_model()
    configure_embedding_model()

    # Create QueryEngineTool for hotel policy and services
    hotel_policy_and_services_tool = QueryEngineTool.from_defaults(
        hotel_query_engine(),
        name="Hotel",
        description= HOTEL_TOOL_DESCRIPTION
    )

    gujarat_wiki_tool = QueryEngineTool.from_defaults(
        gujarat_query_engine(),
        name="Gujarat",
        description="This tool provides information about the state of Gujarat, including popular tourist destinations, "
    )

    bihar_wiki_tool = QueryEngineTool.from_defaults(
        bihar_query_engine(),
        name="Bihar",
        description="This tool provides information about the state of Bihar, including popular tourist destinations, "
    )

    # Create FunctionTool for hotel reservation availability
    reservation_availability_tool = FunctionTool.from_defaults(fn=check_hotel_reservation_availability)

    # Create the ReActAgent with the tools
    return ReActAgent.from_tools(
        [reservation_availability_tool, hotel_policy_and_services_tool, gujarat_wiki_tool, bihar_wiki_tool],
        llm=Settings.llm,
        verbose=True
    )


def isCollectionExists(collection_name: str) -> bool:
    db = chromadb.PersistentClient(path="./src/chroma_db")
    collections = db.list_collections()
    for collection in collections:
        if collection.name == collection_name:
            return True
    return False