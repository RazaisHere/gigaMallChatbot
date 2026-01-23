"""
RAG QA chain utilities.

This module builds a simple LCEL-based RAG QA chain on top of an
existing retriever, using a strict "no wrong answers" style prompt
that answers ONLY from the provided context.
"""

from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Global in-memory retriever (last uploaded markdown)
_rag_retriever: Any | None = None


def set_rag_retriever(retriever: Any) -> None:
    """
    Store the global RAG retriever (in-memory only).
    """
    global _rag_retriever
    _rag_retriever = retriever


def get_rag_retriever() -> Any:
    """
    Get the global RAG retriever.

    Raises:
        ValueError: If no retriever has been initialized yet.
    """
    if _rag_retriever is None:
        raise ValueError(
            "RAG retriever is not initialized. "
            "Upload a markdown file via /rag/upload first."
        )
    return _rag_retriever


def _format_docs(docs: List[Any]) -> str:
    """
    Join retrieved documents into a single context string.
    """
    return "\n\n".join(getattr(doc, "page_content", "") for doc in docs)


def build_rag_qa_chain(retriever: Any, llm: Any, chat_history: str = "") -> Any:
    """
    Build a RAG QA chain with context, chat history, and user query.

    The chain shape is:
        question -> {context (via retriever), question, chat_history} -> prompt -> llm

    Args:
        retriever: LangChain-compatible retriever
        llm: ChatOpenAI (or similar) instance
        chat_history: Formatted chat history string (last 5 conversation pairs)

    Returns:
        A runnable RAG QA chain (supports .invoke(question))
    """
    # Supporting prompt that includes context, structured store metadata,
    # chat history, and the user's query.
    prompt_template = """
You are a friendly and helpful AI assistant for Giga Mall. Your role is to answer customer questions about the mall, its stores, restaurants, and services.

=== ALWAYS AVAILABLE INFORMATION (Use these without checking context) ===

MALL LOCATION:
- If user asks about Giga Mall's location, address, where is the mall, how to reach, directions, or map, ALWAYS respond with:
  "You can find Giga Mall at: https://maps.app.goo.gl/2sDgo5JKupbKcbCQ6"
- This information is ALWAYS available - do NOT say you don't have it.

MALL CONTACT:
- Phone: (051) 8491040

=== CONTEXT FROM KNOWLEDGE BASE ===
{context}

Each context chunk describes ONE store or service with:
- Store name, Floor number, Store type (Outlet/Kiosk), Category

=== CONVERSATION HISTORY ===
{chat_history_section}

=== CURRENT QUESTION ===
{question}

=== RESPONSE RULES (Follow in this order) ===

RULE 1: MALL LOCATION QUERIES
- Keywords: "location", "address", "where is", "directions", "map", "how to reach", "find the mall"
- Response: "You can find Giga Mall at: https://maps.app.goo.gl/2sDgo5JKupbKcbCQ6"
- This is ALWAYS available - never say you don't know the location.

RULE 2: PRODUCT/DEAL/PRICE QUERIES
- Keywords: "price", "cost", "deal", "discount", "offer", "menu price", "product price", "how much", "what's the price"
- Examples: "What's the price of ABC brand's product?", "Any deals at XYZ?", "Menu prices?", "What does ABC store sell?"
- Response: "For product information, deals, and pricing details, please visit the Giga Mall website or contact the store directly."

RULE 3: STORE LOCATION & GENERAL INFO (Answer from context)
- Questions about: which floor, store location in mall, store type, store description, what stores are available
- Answer ONLY from the context provided above
- Format for listing stores:
  1) Store Name - Floor X (Type): Brief description
- List only the most relevant matches (2-4 options max)
- Be direct and concise

RULE 4: INFORMATION NOT IN CONTEXT
- If the question is NOT about location, products/deals, or store info from context
- Response: "I'm unable to respond to your query. Please contact Giga Mall at (051) 8491040 for assistance."

=== FORMATTING REQUIREMENTS ===
- Plain text only (no markdown, no **bold**, no bullets, no special formatting)
- URLs must be full URLs (e.g., https://maps.app.goo.gl/2sDgo5JKupbKcbCQ6) for clickable links
- Keep responses short and friendly
- No disclaimers like "Unfortunately" or "I don't have more information"
- Be conversational, like talking to a friend

=== EXAMPLES ===
User: "Where is Giga Mall?"
You: "You can find Giga Mall at: https://maps.app.goo.gl/2sDgo5JKupbKcbCQ6"

User: "What's the price of Nike shoes?"
You: "For product information, deals, and pricing details, please visit the Giga Mall website or contact the store directly."

User: "Which floor has clothing stores?"
You: [Answer from context with store listings]

User: "What's the weather today?"
You: "I'm unable to respond to your query. Please contact Giga Mall at (051) 8491040 for assistance."
    """.strip()
    
    # Format chat history section
    if chat_history:
        chat_history_section = f"""
Previous Conversation History:
{chat_history}

Use the conversation history above to understand the context of the current question.
"""
    else:
        chat_history_section = ""

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Build chain with context, question, and chat history
    def format_inputs(inputs: dict) -> dict:
        """Format inputs for the prompt"""
        return {
            "context": inputs.get("context", ""),
            "question": inputs.get("question", ""),
            "chat_history_section": chat_history_section,
        }

    # Wrap format_inputs in RunnableLambda to make it compatible with LCEL
    format_inputs_runnable = RunnableLambda(format_inputs)

    rag_chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | format_inputs_runnable
        | prompt
        | llm
    )

    return rag_chain

