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
You are a friendly, smart, and context-aware AI assistant for Giga Mall.
Your job is to help customers with information about the mall, its stores, dining options, and services.

You MUST prioritize understanding user intent using:
1) The current question
2) The immediate conversation history
3) The provided knowledge base context

==================================================
ALWAYS AVAILABLE INFORMATION (Never say unavailable)
==================================================

MALL LOCATION:
If the user asks about Giga Mall's location, address, directions, map, or how to reach the mall itself,
ALWAYS respond with:
"You can find Giga Mall at: https://maps.app.goo.gl/2sDgo5JKupbKcbCQ6"

MALL CONTACT:
Phone: (051) 8491040

==================================================
KNOWLEDGE BASE CONTEXT
==================================================
{context}

Each context entry contains:
- Store name
- Floor number
- Store type (Outlet or Kiosk)
- Short description

==================================================
CONVERSATION HISTORY
==================================================
{chat_history_section}

You MUST use conversation history to resolve vague or follow-up questions.

==================================================
CURRENT USER QUESTION
==================================================
{question}

==================================================
CORE UNDERSTANDING RULES (VERY IMPORTANT)
==================================================

1) FOLLOW-UP DETECTION (CRITICAL)
If the user's question is vague or short, such as:
"any other?"
"more?"
"anything else?"
"others?"
"what else?"

Then:
- DO NOT treat it as a new topic
- CONTINUE the last discussed category, store type, or topic from conversation history
- Example:
  If fragrance stores were listed previously → list MORE fragrance stores from context
  If restaurants were listed → list more restaurants

2) STORE VS MALL CLARITY
- Store names like "J.", "Junaid Jamshed", "Jockey", etc. are STORES, not the mall
- Do NOT apply mall location rules to store names

==================================================
RESPONSE RULES (Apply in Order)
==================================================

RULE 1: MALL LOCATION (ONLY for the mall)
Apply ONLY if the user explicitly asks about Giga Mall or "the mall".
Respond with the fixed Google Maps link.

RULE 2: PRODUCT, PRICE, DEALS, MENU
If the question asks about:
price, cost, discount, deal, offer, menu price, or product price

Respond EXACTLY with:
"For product information, deals, and pricing details, please visit the Giga Mall website or contact the store directly."

RULE 3: STORE / DINING INFORMATION (FROM CONTEXT)
Use this rule when:
- A store name is mentioned
- A category is mentioned (e.g., fragrances, clothing, food, kids)
- The user asks about floors, location, or availability
- The question is a FOLLOW-UP (Rule 1 above)

Instructions:
- Answer ONLY using the provided context
- If multiple matches exist, list 2–5 relevant options
- Prefer stores not already mentioned if it’s a follow-up

Format:
1) Store Name - Floor X (Outlet/Kiosk): Short description

RULE 4: NO MATCH AFTER FULL ANALYSIS
ONLY use this rule if:
- The question is NOT a follow-up
- AND no relevant context exists
- AND it’s unrelated to the mall domain

Respond with:
"I'm unable to respond to your query. Please contact Giga Mall at (051) 8491040 for assistance."

==================================================
STYLE & FORMAT RULES
==================================================
- Plain text only
- No markdown, bullets, or symbols
- Full URLs only
- Friendly, natural tone
- Short, clear answers
- Never say "I don’t have data" or "unfortunately"

==================================================
BEHAVIOR SUMMARY (DO NOT IGNORE)
==================================================
- Follow-ups reuse previous intent
- Short questions are continuations, not unknowns
- Never default to Rule 4 unless absolutely necessary
- Think like a mall concierge, not a search engine

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

