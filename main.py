"""
Mall Chatbot API
Main FastAPI application with PostgreSQL conversation history
"""

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any
import asyncio
import gc
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from chat_history import ChatHistoryManager
from database import get_db
from services.llm import get_llm
from services.rag import build_retriever_from_markdown, load_existing_retriever
from services.rag_chain import set_rag_retriever, build_rag_qa_chain

# ============================================================================
# Logging Configuration
# ============================================================================
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "chat_history.log"),
        logging.StreamHandler()
    ]
)

# Create separate log file for RAG workflow
rag_logger = logging.getLogger("rag_workflow")
rag_file_handler = logging.FileHandler(log_dir / "rag_workflow.log")
rag_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
rag_logger.addHandler(rag_file_handler)
rag_logger.setLevel(logging.INFO)

history_logger = logging.getLogger("chat_history")
logger = logging.getLogger(__name__)

# ============================================================================
# FastAPI App Setup
# ============================================================================
app = FastAPI(title="Mall Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://192.168.18.60:3000",
        "https://the-giga-mall-web-git-feature-chatbot-gigas-projects-ccabf899.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ============================================================================
# Request/Response Models
# ============================================================================
class RagChatRequest(BaseModel):
    """RAG chat request model"""
    message: str
    session_id: str | None = None


class RagChatResponse(BaseModel):
    """RAG chat response model"""
    answer: str
    session_id: str


# ============================================================================
# Helper Functions
# ============================================================================
def get_or_create_session_id(request_session_id: str | None) -> str:
    """Generate or return existing session ID"""
    return request_session_id or str(uuid.uuid4())


# ============================================================================
# API Endpoints
# ============================================================================
@app.get("/")
def root():
    """Root endpoint - health check"""
    return {"message": "Mall Chatbot API is running!"}


@app.post("/rag/upload")
async def upload_markdown(file: UploadFile = File(...)):
    """
    Upload a markdown file and build a basic RAG retriever.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only markdown (.md) files are supported.")

    # Delete all previous uploaded files
    file_uploaded_dir = Path("fileUploaded")
    if file_uploaded_dir.exists():
        for old_file in file_uploaded_dir.glob("*"):
            if old_file.is_file():
                old_file.unlink()
                logger.info(f"Deleted previous uploaded file: {old_file}")
    else:
        file_uploaded_dir.mkdir(exist_ok=True)

    # Store new file in fileUploaded folder
    file_path = file_uploaded_dir / f"{uuid.uuid4()}_{filename}"
    contents = await file.read()
    file_path.write_bytes(contents)
    logger.info(f"Saved new markdown file: {file_path}")

    # Run complete RAG pipeline
    retriever = build_retriever_from_markdown(str(file_path))
    set_rag_retriever(retriever)
    logger.info("RAG pipeline completed - new ChromaDB created")

    return {
        "message": "Markdown file processed and RAG retriever created successfully.",
        "file_path": str(file_path),
        "status": "RAG pipeline completed"
    }


def stream_rag_response(request: RagChatRequest, db: Session):
    """
    Stream RAG chat response word by word using Server-Sent Events (SSE)
    
    Complete RAG Pipeline with Streaming:
    1. Load existing Chroma DB retriever
    2. Convert user query to embeddings (automatic via retriever)
    3. Perform similarity search in Chroma vector store
    4. Retrieve related chunks (top k=5)
    5. Get chat history (last 5 conversation pairs)
    6. Build prompt with: context (retrieved chunks) + user query + chat history
    7. Stream response using LLM
    
    Yields SSE-formatted data chunks for real-time streaming to frontend
    """
    session_id = get_or_create_session_id(request.session_id)
    
    # Load existing Chroma DB retriever (not create new one)
    # This automatically:
    # 1. Converts query to embeddings
    # 2. Performs similarity search in vector store
    # 3. Retrieves top k=5 relevant chunks
    try:
        retriever = load_existing_retriever()
    except ValueError as e:
        error_msg = f"RAG system not initialized: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'data': error_msg})}\n\n"
        return
    
    # Get chat history (last 5 conversation pairs)
    history_messages = ChatHistoryManager.get_last_conversation_pairs(
        db, session_id, num_pairs=5
    )
    
    # Format chat history as string for prompt
    chat_history_str = ""
    if history_messages:
        chat_history_lines = []
        for msg in history_messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            chat_history_lines.append(f"{role_label}: {msg['message']}")
        chat_history_str = "\n".join(chat_history_lines)
    
    # Log user query
    rag_logger.info("\n" + "=" * 80)
    rag_logger.info("NEW USER QUERY RECEIVED")
    rag_logger.info("=" * 80)
    rag_logger.info(f"Session ID: {session_id}")
    rag_logger.info(f"User Query: {request.message}")
    rag_logger.info(f"Chat History Available: {'Yes' if chat_history_str else 'No'}")
    if chat_history_str:
        rag_logger.info(f"Chat History:\n{chat_history_str}")
    rag_logger.info("=" * 80)
    
    # Save user message to database
    ChatHistoryManager.save_message(
        db=db,
        session_id=session_id,
        role="user",
        message=request.message
    )
    
    # Initialize streaming LLM for RAG
    streaming_llm = get_llm(streaming=True)
    
    # Build RAG chain with chat history
    rag_chain = build_rag_qa_chain(retriever, streaming_llm, chat_history_str)
    
    full_response = ""
    
    try:
        # Send session ID first
        yield f"data: {json.dumps({'type': 'session_id', 'data': session_id})}\n\n"
        
        # Stream RAG chain response chunk by chunk
        # The chain automatically: retrieves context -> formats prompt -> streams LLM response
        for chunk in rag_chain.stream(request.message):
            if hasattr(chunk, 'content') and chunk.content:
                content = chunk.content
                full_response += content
                yield f"data: {json.dumps({'type': 'content', 'data': content})}\n\n"
            elif isinstance(chunk, str):
                full_response += chunk
                yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
        
        # Send completion signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        # Log final response
        rag_logger.info("\n" + "=" * 80)
        rag_logger.info("FINAL RESPONSE GENERATED:")
        rag_logger.info("=" * 80)
        rag_logger.info(full_response)
        rag_logger.info("=" * 80 + "\n")
        
        # Save assistant response to database
        ChatHistoryManager.save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            message=full_response
        )
        
    except Exception as e:
        error_msg = "I'm having trouble processing your request. Please try again."
        yield f"data: {json.dumps({'type': 'error', 'data': error_msg})}\n\n"
        
        # Save error message to database
        ChatHistoryManager.save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            message=error_msg
        )


@app.post("/rag/chat", response_model=RagChatResponse)
async def rag_chat(request: RagChatRequest, db: Session = Depends(get_db)):
    """
    Simple RAG chat endpoint - returns complete response (non-streaming)
    
    Complete RAG Pipeline:
    - Loads existing Chroma DB retriever
    - Converts query to embeddings and performs similarity search
    - Retrieves top k=5 relevant chunks
    - Gets chat history (last 5 conversation pairs)
    - Builds prompt with context + query + history
    - Returns complete response using LLM
    
    Use this endpoint for Postman testing or non-streaming clients.
    """
    session_id = get_or_create_session_id(request.session_id)
    
    # Load existing Chroma DB retriever (not create new one)
    # This automatically:
    # 1. Converts query to embeddings
    # 2. Performs similarity search in vector store
    # 3. Retrieves top k=5 relevant chunks
    try:
        retriever = load_existing_retriever()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Get chat history (last 5 conversation pairs)
    history_messages = ChatHistoryManager.get_last_conversation_pairs(
        db, session_id, num_pairs=5
    )
    
    # Format chat history as string for prompt
    chat_history_str = ""
    if history_messages:
        chat_history_lines = []
        for msg in history_messages:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            chat_history_lines.append(f"{role_label}: {msg['message']}")
        chat_history_str = "\n".join(chat_history_lines)
    
    # Log user query
    rag_logger.info("\n" + "=" * 80)
    rag_logger.info("NEW USER QUERY RECEIVED (NON-STREAMING)")
    rag_logger.info("=" * 80)
    rag_logger.info(f"Session ID: {session_id}")
    rag_logger.info(f"User Query: {request.message}")
    rag_logger.info(f"Chat History Available: {'Yes' if chat_history_str else 'No'}")
    if chat_history_str:
        rag_logger.info(f"Chat History:\n{chat_history_str}")
    rag_logger.info("=" * 80)
    
    # Save user message to database
    ChatHistoryManager.save_message(
        db=db,
        session_id=session_id,
        role="user",
        message=request.message
    )
    
    # Non-streaming LLM for RAG
    rag_llm = get_llm(streaming=False)
    
    # Build RAG chain with chat history
    rag_chain = build_rag_qa_chain(retriever, rag_llm, chat_history_str)
    
    try:
        # Invoke chain: automatically does embeddings -> similarity search -> retrieval
        result: Any = rag_chain.invoke(request.message)
        # ChatOpenAI returns an AIMessage; fall back to string for safety
        answer = getattr(result, "content", str(result))
        
        # Log final response
        rag_logger.info("\n" + "=" * 80)
        rag_logger.info("FINAL RESPONSE GENERATED (NON-STREAMING):")
        rag_logger.info("=" * 80)
        rag_logger.info(answer)
        rag_logger.info("=" * 80 + "\n")
    except Exception as e:
        error_msg = "I'm having trouble processing your request. Please try again."
        # Save error message to database
        ChatHistoryManager.save_message(
            db=db,
            session_id=session_id,
            role="assistant",
            message=error_msg
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate RAG answer: {str(e)}",
        )
    
    # Save assistant response to database
    ChatHistoryManager.save_message(
        db=db,
        session_id=session_id,
        role="assistant",
        message=answer
    )
    
    return RagChatResponse(
        answer=answer,
        session_id=session_id
    )


@app.post("/rag/stream")
async def rag_stream(request: RagChatRequest, db: Session = Depends(get_db)):
    """
    Streaming RAG chat endpoint - streams response word by word
    
    Complete RAG Pipeline with Streaming:
    - Loads existing Chroma DB retriever
    - Converts query to embeddings and performs similarity search
    - Retrieves top k=5 relevant chunks
    - Gets chat history (last 5 conversation pairs)
    - Builds prompt with context + query + history
    - Streams response using LLM
    
    Uses Server-Sent Events (SSE) to stream the response in real-time.
    Use this endpoint for frontend applications that need streaming.
    """
    return StreamingResponse(
        stream_rag_response(request, db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
