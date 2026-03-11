"""
SendPulse Social Media Integration Service
Handles OAuth token management and message sending for WhatsApp, Instagram, and Messenger
"""

import re
import logging
import datetime
import requests
from typing import Optional, List, Dict, Any, Union
from config import settings

logger = logging.getLogger(__name__)

# SendPulse token management (in-memory cache)
_sendpulse_token: Dict[str, Any] = {
    "access_token": None,
    "expires_at": 0
}


def get_valid_sendpulse_token() -> Optional[str]:
    """
    Retrieve or renew a valid SendPulse access token.
    Uses OAuth client credentials flow to get access token.
    
    Returns:
        Access token string if successful, None otherwise
    """
    global _sendpulse_token
    
    # Check if token is still valid (with 60 second buffer)
    current_time = datetime.datetime.now().timestamp()
    if _sendpulse_token["access_token"] and current_time < (_sendpulse_token["expires_at"] - 60):
        return _sendpulse_token["access_token"]
    
    # Generate a new token
    url = "https://api.sendpulse.com/oauth/access_token"
    payload = {
        "grant_type": "client_credentials",
        "client_id": settings.sendpulse_client_id,
        "client_secret": settings.sendpulse_client_secret
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        if response.status_code == 200:
            token_data = response.json()
            _sendpulse_token["access_token"] = token_data["access_token"]
            # expires_in is typically in seconds
            _sendpulse_token["expires_at"] = current_time + token_data.get("expires_in", 3600)
            logger.info("Successfully generated new SendPulse access token")
            return token_data["access_token"]
        else:
            logger.error(f"Failed to generate access token: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error generating SendPulse access token: {str(e)}")
        return None


def split_message_smartly(text: str, max_length: int = 512) -> List[str]:
    """
    Splits a long text into chunks without breaking sentences or abbreviations.
    
    Args:
        text: The text to split
        max_length: Maximum length per chunk (default: 512 for WhatsApp)
    
    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_length:
        return [text] if text else []
    
    chunks = []
    
    while len(text) > max_length:
        # Find the last proper sentence-ending punctuation before the limit
        # Avoid breaking on abbreviations like Dr., Mr., Ms., etc.
        match = re.search(
            r'(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bMrs)(?<!\bProf)(?<!\bInc)(?<!\bLtd)\.|\?|!',
            text[:max_length]
        )
        
        if match:
            split_index = match.end()
        else:
            # Try to split at newline
            split_index = text.rfind('\n', 0, max_length)
            if split_index == -1:
                # Last resort: split at max_length
                split_index = max_length
        
        chunks.append(text[:split_index].strip())
        text = text[split_index:].strip()
    
    if text:
        chunks.append(text)
    
    return chunks


async def send_message_to_sendpulse(
    message: Union[str, List[str]],
    service: str,
    contact_id: str
) -> List[Dict[str, Any]]:
    """
    Dispatches a message to the appropriate SendPulse service.
    
    Args:
        message: Message text or list of messages to send
        service: Service type ('whatsapp', 'instagram', 'messenger')
        contact_id: SendPulse contact ID
    
    Returns:
        List of API responses
    """
    if isinstance(message, list):
        all_responses = []
        for item in message:
            if service == "instagram":
                responses = await send_message_to_instagram(contact_id, item)
            elif service == "messenger":
                responses = await send_message_to_messenger(contact_id, item)
            elif service == "whatsapp":
                responses = await send_message_to_whatsapp(contact_id, item)
            else:
                logger.warning(f"Unknown service type: {service}")
                continue
            if responses:
                all_responses.extend(responses)
        return all_responses
    else:
        if not isinstance(message, str):
            message = str(message)
        
        if service == "instagram":
            return await send_message_to_instagram(contact_id, message)
        elif service == "messenger":
            return await send_message_to_messenger(contact_id, message)
        elif service == "whatsapp":
            return await send_message_to_whatsapp(contact_id, message)
        else:
            logger.warning(f"Unknown service type: {service}")
            return []


async def send_message_to_whatsapp(contact_id: str, message: str) -> List[Dict[str, Any]]:
    """
    Send a message back to the user via SendPulse WhatsApp API.
    Messages are split if they exceed 512 characters.
    
    Args:
        contact_id: SendPulse contact ID
        message: Message text to send
    
    Returns:
        List of API responses
    """
    access_token = get_valid_sendpulse_token()
    if not access_token:
        logger.error("Failed to retrieve valid access token for WhatsApp")
        return []
    
    url = "https://api.sendpulse.com/whatsapp/contacts/send"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Split the message smartly without cutting sentences
    message_chunks = split_message_smartly(message, max_length=512)
    responses = []
    
    for chunk in message_chunks:
        payload = {
            "contact_id": contact_id,
            "message": {
                "type": "text",
                "text": {
                    "body": chunk
                }
            }
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response_data = response.json()
            responses.append(response_data)
            logger.info(f"SendPulse WhatsApp Response: {response.status_code} - {response_data}")
        except Exception as e:
            logger.error(f"Error sending WhatsApp message: {str(e)}")
            responses.append({"error": str(e)})
    
    return responses


async def send_message_to_instagram(
    contact_id: str,
    message: str,
    is_image: bool = False
) -> List[Dict[str, Any]]:
    """
    Send a text or image message via SendPulse to Instagram.
    
    Args:
        contact_id: SendPulse contact ID
        message: Message text or image URL(s)
        is_image: Whether the message is an image
    
    Returns:
        List of API responses
    """
    access_token = get_valid_sendpulse_token()
    if not access_token:
        logger.error("Failed to retrieve valid access token for Instagram")
        return []
    
    url = "https://api.sendpulse.com/instagram/contacts/send"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    responses = []
    
    if is_image and isinstance(message, list):
        # Send multiple images
        for image_url in message:
            payload = {
                "contact_id": contact_id,
                "messages": [
                    {
                        "type": "file",
                        "message": {
                            "attachment": {
                                "type": "image",
                                "payload": {"url": image_url}
                            }
                        }
                    }
                ]
            }
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                response_data = response.json()
                responses.append(response_data)
                logger.info(f"SendPulse Instagram Image Response: {response.status_code} - {response_data}")
            except Exception as e:
                logger.error(f"Error sending Instagram image: {str(e)}")
                responses.append({"error": str(e)})
    else:
        # Send text message(s)
        message_chunks = split_message_smartly(message, max_length=512)
        for chunk in message_chunks:
            payload = {
                "contact_id": contact_id,
                "messages": [
                    {
                        "type": "text",
                        "message": {
                            "text": chunk
                        }
                    }
                ]
            }
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                response_data = response.json()
                responses.append(response_data)
                logger.info(f"SendPulse Instagram Text Response: {response.status_code} - {response_data}")
            except Exception as e:
                logger.error(f"Error sending Instagram message: {str(e)}")
                responses.append({"error": str(e)})
    
    return responses


async def send_message_to_messenger(contact_id: str, message: str) -> List[Dict[str, Any]]:
    """
    Send a message back to the user via SendPulse Messenger API.
    
    Args:
        contact_id: SendPulse contact ID
        message: Message text to send
    
    Returns:
        List of API responses
    """
    access_token = get_valid_sendpulse_token()
    if not access_token:
        logger.error("Failed to retrieve valid access token for Messenger")
        return []
    
    url = "https://api.sendpulse.com/messenger/contacts/send"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    message_chunks = split_message_smartly(message, max_length=512)
    responses = []
    
    for chunk in message_chunks:
        payload = {
            "contact_id": contact_id,
            "message": {
                "type": "RESPONSE",
                "tag": "CUSTOMER_FEEDBACK",
                "content_type": "message",
                "text": chunk
            }
        }
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response_data = response.json()
            responses.append(response_data)
            logger.info(f"SendPulse Messenger Response: {response.status_code} - {response_data}")
        except Exception as e:
            logger.error(f"Error sending Messenger message: {str(e)}")
            responses.append({"error": str(e)})
    
    return responses
