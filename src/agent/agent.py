from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool
from config.vector_store_config import initialize_vector_store
from config.llm_config import configure_language_model
from config.embedding_model_config import configure_embedding_model
from llama_index.core import Settings

def check_hotel_reservation_availability(date: str) -> str:
    """Check if hotel reservations are available for the given date."""
    if date.lower() == "tomorrow":
        return "Yes, reservations are available for tomorrow."
    else:
        return "Sorry, reservations are only available for tomorrow at the moment."

def initialize_agent():
    """Define and initialize the ReActAgent with necessary tools."""
    configure_language_model()
    configure_embedding_model()

    # Create QueryEngineTool for hotel policy and services
    hotel_policy_and_services_tool = QueryEngineTool.from_defaults(
        initialize_vector_store(),
        name="Hotel",
        description="Provide detailed information about hotel policies, services, amenities, and general inquiries. "
                    "This includes check-in/check-out times, cancellation policies, available facilities (e.g., gym, pool), "
                    "room service details, parking information, pet policies, and other hotel-related services."
    )


    # Create FunctionTool for hotel reservation availability
    reservation_availability_tool = FunctionTool.from_defaults(fn=check_hotel_reservation_availability)

    # Create the ReActAgent with the tools
    return ReActAgent.from_tools(
        [reservation_availability_tool, hotel_policy_and_services_tool],
        llm=Settings.llm,
        verbose=True
    )
