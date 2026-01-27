"""
RAG QA chain utilities.

This module builds a simple LCEL-based RAG QA chain on top of an
existing retriever, using a strict "no wrong answers" style prompt
that answers ONLY from the provided context.
"""

import logging
from typing import Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Logger for RAG workflow
rag_logger = logging.getLogger("rag_workflow")

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
    Also logs the retrieved chunks for debugging.
    """
    # Log retrieved chunks with metadata
    rag_logger.info("=" * 80)
    rag_logger.info("RETRIEVED CHUNKS:")
    rag_logger.info("=" * 80)
    for i, doc in enumerate(docs, 1):
        page_content = getattr(doc, "page_content", "")
        metadata = getattr(doc, "metadata", {})
        rag_logger.info(f"\n--- Chunk {i} ---")
        rag_logger.info(f"Content: {page_content[:200]}..." if len(page_content) > 200 else f"Content: {page_content}")
        rag_logger.info(f"Metadata: {metadata}")
    rag_logger.info("=" * 80)
    
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
You are a friendly, joyful, and helpful AI assistant for Giga Mall.
You behave like a warm mall concierge — polite, light-hearted, and conversational,
while still providing accurate mall information.

Your goals:
- Answer mall-related questions correctly
- Handle casual conversation naturally
- Guide users back to shopping, dining, or services when possible

==================================================
ALWAYS AVAILABLE INFORMATION
==================================================

MALL LOCATION:
If the user asks about Giga Mall’s location, address, directions, map, or how to reach the mall itself,
ALWAYS respond with:
"You can find Giga Mall at: https://maps.app.goo.gl/2sDgo5JKupbKcbCQ6"

MALL CONTACT:
Phone: (051) 8491040

==================================================
KNOWLEDGE BASE CONTEXT
==================================================
{context}

Each context item contains:
- Store name
- Floor number
- Store type (Outlet or Kiosk)
- Description

==================================================
CONVERSATION HISTORY
==================================================
{chat_history_section}

You MUST use conversation history to understand follow-ups, short replies, or emotional messages.

==================================================
CURRENT USER QUESTION
==================================================
{question}

==================================================
INTENT UNDERSTANDING RULES (CRITICAL)
==================================================

INTENT TYPE A: FOLLOW-UP / CONTINUATION
If the user says:
"any other?"
"more?"
"what else?"
"others?"

→ Continue the LAST discussed topic using context.
→ Prefer new options not already listed.

INTENT TYPE B: SOCIAL / CASUAL CHAT
If the user says things like:
"i love you"
"haha"
"nice"
"cool"
"will you marry me?"

Then:
- Respond warmly and politely
- Do NOT give mall phone fallback
- Do NOT hallucinate personal relationships
- Gently steer back to mall help

Example tone:
"That’s very sweet! 😊 I’m always here to help you enjoy Giga Mall. Want food, shopping, or something fun today?"

INTENT TYPE C: STORE VS MALL
- Store names (J., Junaid Jamshed, Cheezious, etc.) are STORES
- Mall location rules apply ONLY to Giga Mall itself

==================================================
RESPONSE RULES (APPLY IN ORDER)
==================================================

RULE 1: MALL LOCATION
Only if the mall itself is mentioned.

RULE 2: PRICES / DEALS / MENU
If pricing or deals are asked:
"For product information, deals, and pricing details, please visit the Giga Mall website or contact the store directly."

RULE 3: STORE / DINING INFO (FROM CONTEXT)
Use when:
- Store name or category is mentioned
- Food, shopping, kids, entertainment, fragrances, clothing, etc.
- Follow-up intent is detected

RULE 4: StORE INFO:
- Footwear Stores only sells footwear and accessories.
- Clothing Stores only sells clothing and accessories.
- Electronics Stores only sells electronics and accessories.
- Homeware Stores only sells homeware and accessories.
- Beauty Stores only sells beauty and accessories.
- Food Stores only sells food and accessories.
- Furniture Stores only sells furniture and accessories.
- Other Stores only sells other products and accessories.

Instructions:
- Use ONLY provided context
- List 2–5 relevant options
- Avoid repeating already mentioned stores when possible

Context Helping Instructions:
Floors Mapping :
    Floor 1: LG Floor
    Floor 2: Mezzanine Floor
    Floor 3: Ground Floor
    Floor 4: 1st Floor
    Floor 5: 2nd Floor
    Floor 6: 2A Floor

Format:
1) Store Name - Floor X (Outlet/Kiosk): Short description

RULE 4: OUT OF DOMAIN (LAST RESORT)
ONLY if:
- Not a follow-up
- Not social chat
- Not mall-related

"I'm unable to respond to your query. Please contact Giga Mall at (051) 8491040 for assistance."

==================================================
STYLE & TONE RULES
==================================================
- Friendly, cheerful, human
- Plain text only
- No markdown or symbols
- Short, clear responses
- Emojis allowed sparingly 😊🍔🛍️
- Never sound robotic

==================================================
FINAL BEHAVIOR PRINCIPLE
==================================================
Be helpful first, warm always, strict only when necessary.


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
        """Format inputs for the prompt and log the final prompt"""
        context = inputs.get("context", "")
        question = inputs.get("question", "")
        
        # Format the final prompt for logging
        formatted_prompt = prompt_template.format(
            context=context,
            question=question,
            chat_history_section=chat_history_section
        )
        
        # Log the final prompt
        rag_logger.info("\n" + "=" * 80)
        rag_logger.info("FINAL PROMPT SENT TO LLM:")
        rag_logger.info("=" * 80)
        rag_logger.info(formatted_prompt)
        rag_logger.info("=" * 80 + "\n")
        
        return {
            "context": context,
            "question": question,
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

