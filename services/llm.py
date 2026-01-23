"""
LLM Service for OpenAI Chat
"""

import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def get_llm(streaming: bool = False):
    """
    Get OpenAI Chat LLM instance with gpt-4o-mini model
    
    Args:
        streaming: If True, enables streaming mode
    
    Returns:
        ChatOpenAI instance configured with gpt-4o-mini
    """
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,  # LOW creativity (important)
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        streaming=streaming,
    )
