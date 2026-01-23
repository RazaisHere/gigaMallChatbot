"""
Chat History Management
Manages conversation history in PostgreSQL
"""

from sqlalchemy.orm import Session
from database import ChatMessage
from typing import List, Dict
from datetime import datetime, timedelta


class ChatHistoryManager:
    """Manages conversation history in PostgreSQL"""
    
    @staticmethod
    def save_message(
        db: Session,
        session_id: str,
        role: str,
        message: str
    ) -> ChatMessage:
        """Save a message to database"""
        chat_msg = ChatMessage(
            session_id=session_id,
            role=role,
            message=message
        )
        db.add(chat_msg)
        db.commit()
        db.refresh(chat_msg)
        return chat_msg
    
    @staticmethod
    def get_recent_history(
        db: Session,
        session_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent conversation history"""
        messages = db.query(ChatMessage)\
            .filter(ChatMessage.session_id == session_id)\
            .order_by(ChatMessage.timestamp.desc())\
            .limit(limit)\
            .all()
        
        # Reverse to get chronological order
        messages = list(reversed(messages))
        
        return [
            {
                "role": msg.role,
                "message": msg.message,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in messages
        ]
    
    @staticmethod
    def get_last_conversation_pairs(
        db: Session,
        session_id: str,
        num_pairs: int = 5
    ) -> List[Dict]:
        """
        Get last N conversation pairs (user + assistant = 1 pair)
        
        A conversation pair consists of:
        - User message (question)
        - Assistant message (reply)
        
        Args:
            db: Database session
            session_id: Session ID to get history for
            num_pairs: Number of conversation pairs to retrieve (default: 5)
        
        Returns:
            List of messages in chronological order (last N pairs, flattened)
        """
        # Get enough messages to ensure we have num_pairs pairs
        # Need at least 2 * num_pairs messages, plus buffer for incomplete pairs
        messages = db.query(ChatMessage)\
            .filter(ChatMessage.session_id == session_id)\
            .order_by(ChatMessage.timestamp.asc())\
            .limit(num_pairs * 2 + 2)\
            .all()
        
        if not messages:
            return []
        
        # Convert to dict format
        message_dicts = [
            {
                "role": msg.role,
                "message": msg.message,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in messages
        ]
        
        # Group messages into pairs (user -> assistant)
        pairs = []
        i = 0
        while i < len(message_dicts) - 1:
            current = message_dicts[i]
            next_msg = message_dicts[i + 1]
            
            # Check if we have a complete pair (user followed by assistant)
            if current["role"] == "user" and next_msg["role"] == "assistant":
                pairs.append([current, next_msg])
                i += 2  # Skip both messages
            else:
                i += 1  # Skip current message
        
        # Take the last num_pairs pairs
        pairs = pairs[-num_pairs:] if len(pairs) > num_pairs else pairs
        
        # Flatten pairs back into individual messages
        result = []
        for pair in pairs:
            result.extend(pair)
        
        return result
    
    @staticmethod
    def get_conversation_context(
        db: Session,
        session_id: str,
        last_n: int = 5
    ) -> str:
        """Get formatted conversation context for AI"""
        messages = ChatHistoryManager.get_recent_history(db, session_id, last_n)
        
        if not messages:
            return ""
        
        context = "Previous conversation:\n"
        for msg in messages:
            context += f"{msg['role'].title()}: {msg['message']}\n"
        
        return context
    
    @staticmethod
    def clear_old_sessions(db: Session, days: int = 30):
        """Clean up old conversation history"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        db.query(ChatMessage)\
            .filter(ChatMessage.timestamp < cutoff_date)\
            .delete()
        db.commit()
