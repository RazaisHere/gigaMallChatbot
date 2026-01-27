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


def _detect_cuisine_intent(question: str) -> str:
    """
    Detect specific cuisine intent from user question.
    Returns a cuisine hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "arabic foods" -> hint to only show Arabic cuisine restaurants
    - "chinese food" -> hint to only show Chinese food restaurants
    - "desi food" -> hint to only show Desi/Pakistani food restaurants
    """
    question_lower = question.lower()
    
    # Arabic cuisine intent
    if any(keyword in question_lower for keyword in ["arabic food", "arabic foods", "arabic cuisine", "mandi"]):
        return (
            "CUISINE INTENT: User is specifically asking for ARABIC cuisine. "
            "ONLY list restaurants whose context or tags explicitly mention Arabic cuisine, Arabic food, or mandi. "
            "DO NOT list purely Pakistani/desi restaurants as Arabic. "
            "If a restaurant mentions both Arabic and desi (e.g., 'Arabic and desi-inspired'), it is acceptable."
        )
    
    # Chinese cuisine intent
    if any(keyword in question_lower for keyword in ["chinese food", "chinese foods", "chinese cuisine", "chinese restaurant"]):
        return (
            "CUISINE INTENT: User is specifically asking for CHINESE cuisine. "
            "ONLY list restaurants whose context or tags explicitly mention Chinese food, Chinese cuisine, or Chinese dishes. "
            "DO NOT list other Asian cuisines unless they specifically mention Chinese food."
        )
    
    # Desi/Pakistani cuisine intent
    if any(keyword in question_lower for keyword in ["desi food", "desi foods", "pakistani food", "pakistani cuisine", "traditional pakistani"]):
        return (
            "CUISINE INTENT: User is specifically asking for DESI/PAKISTANI cuisine. "
            "ONLY list restaurants whose context or tags explicitly mention desi food, Pakistani food, or traditional Pakistani dishes. "
            "DO NOT list Arabic-only restaurants unless they also mention desi/Pakistani cuisine."
        )
    
    return ""


def _detect_beauty_intent(question: str) -> str:
    """
    Detect specific beauty/fragrance intent from user question.
    Returns a beauty hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "perfumes" -> hint to only show fragrance stores
    - "makeup" -> hint to only show cosmetics stores
    - "skincare" -> hint to only show skincare stores
    - "men's grooming" -> hint to only show men's grooming stores
    """
    question_lower = question.lower()
    
    # Fragrance/Perfume intent
    if any(keyword in question_lower for keyword in ["perfume", "perfumes", "fragrance", "fragrances", "attar", "attars", "cologne", "colognes", "scent", "scents"]):
        return (
            "BEAUTY INTENT: User is specifically asking for FRAGRANCES/PERFUMES. "
            "ONLY list stores whose context or tags explicitly mention perfumes, fragrances, attars, colognes, or scents. "
            "DO NOT list stores that only sell cosmetics or skincare unless they also mention fragrances/perfumes."
        )
    
    # Cosmetics/Makeup intent
    if any(keyword in question_lower for keyword in ["makeup", "cosmetics", "foundation", "lipstick", "mascara", "nail polish", "cosmetic"]):
        return (
            "BEAUTY INTENT: User is specifically asking for COSMETICS/MAKEUP. "
            "ONLY list stores whose context or tags explicitly mention makeup, cosmetics, foundation, lipstick, mascara, or nail polish. "
            "DO NOT list stores that only sell fragrances or skincare unless they also mention cosmetics/makeup."
        )
    
    # Skincare intent
    if any(keyword in question_lower for keyword in ["skincare", "skin care", "beauty products", "herbal", "natural", "organic", "haircare", "hair care"]):
        return (
            "BEAUTY INTENT: User is specifically asking for SKINCARE/BEAUTY PRODUCTS. "
            "ONLY list stores whose context or tags explicitly mention skincare, beauty products, herbal, natural, organic, or haircare. "
            "DO NOT list stores that only sell fragrances or cosmetics unless they also mention skincare/beauty products."
        )
    
    # Men's Grooming intent
    if any(keyword in question_lower for keyword in ["men's grooming", "mens grooming", "grooming", "beard", "beard oil", "men's products"]):
        return (
            "BEAUTY INTENT: User is specifically asking for MEN'S GROOMING PRODUCTS. "
            "ONLY list stores whose context or tags explicitly mention men's grooming, grooming products, beard oils, or men's personal care. "
            "DO NOT list stores that only sell women's cosmetics or fragrances unless they also mention men's grooming products."
        )
    
    # Oriental/Traditional Fragrances intent
    if any(keyword in question_lower for keyword in ["oriental", "arabic attar", "arabic attars", "oud", "musk", "traditional arabic"]):
        return (
            "BEAUTY INTENT: User is specifically asking for ORIENTAL/TRADITIONAL FRAGRANCES. "
            "ONLY list stores whose context or tags explicitly mention oriental fragrances, Arabic attars, oud, musk, or traditional Arabic scents. "
            "DO NOT list stores that only sell western-style designer perfumes unless they also mention oriental/traditional fragrances."
        )
    
    # Western/Designer Fragrances intent
    if any(keyword in question_lower for keyword in ["western perfume", "western fragrances", "designer perfume", "designer perfumes", "designer fragrance"]):
        return (
            "BEAUTY INTENT: User is specifically asking for WESTERN/DESIGNER FRAGRANCES. "
            "ONLY list stores whose context or tags explicitly mention western-style perfumes, designer perfumes, or designer fragrances. "
            "DO NOT list stores that only sell traditional oriental attars unless they also mention western/designer fragrances."
        )
    
    return ""


def _detect_sports_intent(question: str) -> str:
    """
    Detect specific sports/activewear intent from user question.
    Returns a sports hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "sportswear" -> hint to only show sportswear stores
    - "football" -> hint to only show football-related stores
    - "gym gear" -> hint to only show gym/fitness stores
    - "athletic footwear" -> hint to only show athletic footwear stores
    """
    question_lower = question.lower()
    
    # Sportswear/Activewear intent
    if any(keyword in question_lower for keyword in ["sportswear", "sports wear", "activewear", "active wear", "athletic apparel", "athletic wear"]):
        return (
            "SPORTS INTENT: User is specifically asking for SPORTSWEAR/ACTIVEWEAR. "
            "ONLY list stores whose context or tags explicitly mention sportswear, activewear, or athletic apparel. "
            "DO NOT list stores that only sell sports equipment or accessories unless they also mention sportswear/activewear."
        )
    
    # Athletic Footwear intent
    if any(keyword in question_lower for keyword in ["athletic footwear", "sports shoes", "sneakers", "sports footwear", "running shoes", "football boots", "soccer cleats"]):
        return (
            "SPORTS INTENT: User is specifically asking for ATHLETIC FOOTWEAR/SPORTS SHOES. "
            "ONLY list stores whose context or tags explicitly mention athletic footwear, sports shoes, sneakers, or running shoes. "
            "DO NOT list stores that only sell sportswear or equipment unless they also mention footwear/shoes."
        )
    
    # Sports Equipment intent
    if any(keyword in question_lower for keyword in ["sports equipment", "sports gear", "football kits", "jerseys", "tracksuits", "sports accessories"]):
        return (
            "SPORTS INTENT: User is specifically asking for SPORTS EQUIPMENT/GEAR. "
            "ONLY list stores whose context or tags explicitly mention sports equipment, sports gear, football kits, jerseys, or tracksuits. "
            "DO NOT list stores that only sell sportswear or footwear unless they also mention equipment/gear."
        )
    
    # Specific Sports intent
    if any(keyword in question_lower for keyword in ["football", "soccer"]):
        return (
            "SPORTS INTENT: User is specifically asking for FOOTBALL/SOCCER products. "
            "ONLY list stores whose context or tags explicitly mention football, soccer, football kits, football boots, or club jerseys. "
            "DO NOT list stores that only sell other sports products unless they also mention football/soccer."
        )
    
    if any(keyword in question_lower for keyword in ["cricket", "cricket gear", "cricket equipment"]):
        return (
            "SPORTS INTENT: User is specifically asking for CRICKET products. "
            "ONLY list stores whose context or tags explicitly mention cricket, cricket gear, or cricket equipment. "
            "DO NOT list stores that only sell other sports products unless they also mention cricket."
        )
    
    if any(keyword in question_lower for keyword in ["tennis", "tennis equipment", "tennis gear"]):
        return (
            "SPORTS INTENT: User is specifically asking for TENNIS products. "
            "ONLY list stores whose context or tags explicitly mention tennis, tennis equipment, or tennis gear. "
            "DO NOT list stores that only sell other sports products unless they also mention tennis."
        )
    
    if any(keyword in question_lower for keyword in ["basketball", "basketball gear", "basketball equipment"]):
        return (
            "SPORTS INTENT: User is specifically asking for BASKETBALL products. "
            "ONLY list stores whose context or tags explicitly mention basketball, basketball gear, or basketball equipment. "
            "DO NOT list stores that only sell other sports products unless they also mention basketball."
        )
    
    # Gym/Fitness intent
    if any(keyword in question_lower for keyword in ["gym", "gym gear", "gym wear", "fitness", "fitness wear", "training"]):
        return (
            "SPORTS INTENT: User is specifically asking for GYM/FITNESS products. "
            "ONLY list stores whose context or tags explicitly mention gym gear, gym wear, fitness wear, or training equipment. "
            "DO NOT list stores that only sell other sports products unless they also mention gym/fitness."
        )
    
    # Running intent
    if any(keyword in question_lower for keyword in ["running", "running shoes", "running gear", "running equipment"]):
        return (
            "SPORTS INTENT: User is specifically asking for RUNNING products. "
            "ONLY list stores whose context or tags explicitly mention running, running shoes, or running gear. "
            "DO NOT list stores that only sell other sports products unless they also mention running."
        )
    
    return ""


def _detect_electronics_intent(question: str) -> str:
    """
    Detect specific electronics/tech intent from user question.
    Returns an electronics hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "smartphones" -> hint to only show smartphone stores
    - "mobile accessories" -> hint to only show mobile accessory stores
    - "gaming" -> hint to only show gaming stores
    - "smart home" -> hint to only show smart home device stores
    """
    question_lower = question.lower()
    
    # Smartphones/Mobile intent
    if any(keyword in question_lower for keyword in ["smartphone", "smartphones", "mobile", "mobiles", "phone", "phones"]):
        return (
            "ELECTRONICS INTENT: User is specifically asking for SMARTPHONES/MOBILES. "
            "ONLY list stores whose context or tags explicitly mention smartphones, mobiles, or phones. "
            "DO NOT list stores that only sell accessories or other electronics unless they also mention smartphones/mobiles."
        )
    
    # Mobile Accessories intent
    if any(keyword in question_lower for keyword in ["mobile accessories", "phone accessories", "phone cases", "chargers", "cables", "headphones", "mobile cases"]):
        return (
            "ELECTRONICS INTENT: User is specifically asking for MOBILE ACCESSORIES. "
            "ONLY list stores whose context or tags explicitly mention mobile accessories, phone cases, chargers, cables, or headphones. "
            "DO NOT list stores that only sell smartphones or other electronics unless they also mention mobile accessories."
        )
    
    # Gaming intent
    if any(keyword in question_lower for keyword in ["gaming", "gaming consoles", "gaming console", "playstation", "xbox", "nintendo", "video games", "video game", "gaming accessories", "gaming gear"]):
        return (
            "ELECTRONICS INTENT: User is specifically asking for GAMING products. "
            "ONLY list stores whose context or tags explicitly mention gaming, gaming consoles, PlayStation, Xbox, Nintendo, or gaming accessories. "
            "DO NOT list stores that only sell other electronics unless they also mention gaming products."
        )
    
    # Smart Home intent
    if any(keyword in question_lower for keyword in ["smart home", "smart home devices", "air purifier", "air purifiers", "robot vacuum", "robot vacuums"]):
        return (
            "ELECTRONICS INTENT: User is specifically asking for SMART HOME devices. "
            "ONLY list stores whose context or tags explicitly mention smart home devices, air purifiers, robot vacuums, or smart home solutions. "
            "DO NOT list stores that only sell other electronics unless they also mention smart home devices."
        )
    
    # Audio intent
    if any(keyword in question_lower for keyword in ["audio", "audio gear", "premium audio", "headphones", "earphones", "speakers"]):
        return (
            "ELECTRONICS INTENT: User is specifically asking for AUDIO products. "
            "ONLY list stores whose context or tags explicitly mention audio gear, premium audio, headphones, or speakers. "
            "DO NOT list stores that only sell other electronics unless they also mention audio products."
        )
    
    # Electronics/Gadgets general intent
    if any(keyword in question_lower for keyword in ["electronics", "electronic", "gadgets", "gadget", "tech gadgets", "technology"]):
        return (
            "ELECTRONICS INTENT: User is specifically asking for ELECTRONICS/GADGETS. "
            "ONLY list stores whose context or tags explicitly mention electronics, gadgets, or tech gadgets. "
            "DO NOT list stores from other categories unless they also mention electronics/gadgets."
        )
    
    return ""


def _detect_jewelry_intent(question: str) -> str:
    """
    Detect specific jewelry intent from user question.
    Returns a jewelry hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "gold jewelry" -> hint to only show gold jewelry stores
    - "diamond jewelry" -> hint to only show diamond jewelry stores
    - "bridal jewelry" -> hint to only show bridal jewelry stores
    - "silver jewelry" -> hint to only show silver jewelry stores
    """
    question_lower = question.lower()
    
    # Gold Jewelry intent
    if any(keyword in question_lower for keyword in ["gold jewelry", "gold jewellery", "gold", "golden"]):
        return (
            "JEWELRY INTENT: User is specifically asking for GOLD JEWELRY. "
            "ONLY list stores whose context or tags explicitly mention gold jewelry, gold, or traditional gold. "
            "DO NOT list stores that only sell diamond, silver, or other jewelry unless they also mention gold jewelry."
        )
    
    # Diamond Jewelry intent
    if any(keyword in question_lower for keyword in ["diamond jewelry", "diamond jewellery", "diamond", "diamonds", "diamond rings"]):
        return (
            "JEWELRY INTENT: User is specifically asking for DIAMOND JEWELRY. "
            "ONLY list stores whose context or tags explicitly mention diamond jewelry, diamonds, or diamond rings. "
            "DO NOT list stores that only sell gold, silver, or other jewelry unless they also mention diamond jewelry."
        )
    
    # Silver Jewelry intent
    if any(keyword in question_lower for keyword in ["silver jewelry", "silver jewellery", "silver", "silver ornaments"]):
        return (
            "JEWELRY INTENT: User is specifically asking for SILVER JEWELRY. "
            "ONLY list stores whose context or tags explicitly mention silver jewelry, silver ornaments, or silver pieces. "
            "DO NOT list stores that only sell gold, diamond, or other jewelry unless they also mention silver jewelry."
        )
    
    # Bridal Jewelry intent
    if any(keyword in question_lower for keyword in ["bridal jewelry", "bridal jewellery", "bridal sets", "bridal set", "engagement rings", "engagement ring", "bridal wear"]):
        return (
            "JEWELRY INTENT: User is specifically asking for BRIDAL JEWELRY. "
            "ONLY list stores whose context or tags explicitly mention bridal jewelry, bridal sets, engagement rings, or bridal wear. "
            "DO NOT list stores that only sell regular jewelry unless they also mention bridal jewelry."
        )
    
    # Traditional Jewelry intent
    if any(keyword in question_lower for keyword in ["traditional jewelry", "traditional jewellery", "kundan", "kundan work", "handcrafted jewelry", "heritage jewelry"]):
        return (
            "JEWELRY INTENT: User is specifically asking for TRADITIONAL JEWELRY. "
            "ONLY list stores whose context or tags explicitly mention traditional jewelry, Kundan work, handcrafted ornaments, or heritage jewelry. "
            "DO NOT list stores that only sell contemporary or modern jewelry unless they also mention traditional jewelry."
        )
    
    # Contemporary/Modern Jewelry intent
    if any(keyword in question_lower for keyword in ["contemporary jewelry", "contemporary jewellery", "modern jewelry", "high-fashion jewelry"]):
        return (
            "JEWELRY INTENT: User is specifically asking for CONTEMPORARY/MODERN JEWELRY. "
            "ONLY list stores whose context or tags explicitly mention contemporary jewelry, modern designs, or high-fashion jewelry. "
            "DO NOT list stores that only sell traditional jewelry unless they also mention contemporary/modern jewelry."
        )
    
    # Crystal Jewelry intent
    if any(keyword in question_lower for keyword in ["crystal jewelry", "crystal jewellery", "crystal", "swarovski"]):
        return (
            "JEWELRY INTENT: User is specifically asking for CRYSTAL JEWELRY. "
            "ONLY list stores whose context or tags explicitly mention crystal jewelry, crystal creations, or Swarovski. "
            "DO NOT list stores that only sell gold, diamond, or silver jewelry unless they also mention crystal jewelry."
        )
    
    # Gemstone Jewelry intent
    if any(keyword in question_lower for keyword in ["gemstone jewelry", "gemstone jewellery", "gemstone", "gemstones", "semi-precious"]):
        return (
            "JEWELRY INTENT: User is specifically asking for GEMSTONE JEWELRY. "
            "ONLY list stores whose context or tags explicitly mention gemstone jewelry, gemstones, or semi-precious jewelry. "
            "DO NOT list stores that only sell other types of jewelry unless they also mention gemstones."
        )
    
    # General Jewelry intent
    if any(keyword in question_lower for keyword in ["jewelry", "jewellery", "jeweler", "jeweller", "jewels"]):
        return (
            "JEWELRY INTENT: User is asking for JEWELRY in general. "
            "List all jewelry stores from the context that match the user's query. "
            "Include stores that sell gold, diamond, silver, bridal, traditional, or contemporary jewelry."
        )
    
    return ""


def _detect_watches_intent(question: str) -> str:
    """
    Detect specific watches intent from user question.
    Returns a watches hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "watches" -> hint to only show watch stores
    - "luxury watches" -> hint to only show luxury watch stores
    - "sports watches" -> hint to only show sports watch stores
    """
    question_lower = question.lower()
    
    # Luxury Watches intent
    if any(keyword in question_lower for keyword in ["luxury watch", "luxury watches", "premium watch", "high-end watch", "swiss watch"]):
        return (
            "WATCHES INTENT: User is specifically asking for LUXURY/PREMIUM WATCHES. "
            "ONLY list stores whose context or tags explicitly mention luxury watches, premium watches, or Swiss watches. "
            "DO NOT list stores that only sell affordable or budget watches unless they also mention luxury watches."
        )
    
    # Sports Watches intent
    if any(keyword in question_lower for keyword in ["sports watch", "sports watches", "chronograph", "chronographs"]):
        return (
            "WATCHES INTENT: User is specifically asking for SPORTS WATCHES. "
            "ONLY list stores whose context or tags explicitly mention sports watches, chronographs, or sports-inspired timepieces. "
            "DO NOT list stores that only sell dress watches or formal watches unless they also mention sports watches."
        )
    
    # Dress/Formal Watches intent
    if any(keyword in question_lower for keyword in ["dress watch", "dress watches", "formal watch", "formal watches", "elegant watch"]):
        return (
            "WATCHES INTENT: User is specifically asking for DRESS/FORMAL WATCHES. "
            "ONLY list stores whose context or tags explicitly mention dress watches, formal watches, or elegant timepieces. "
            "DO NOT list stores that only sell sports watches unless they also mention dress/formal watches."
        )
    
    # General Watches intent
    if any(keyword in question_lower for keyword in ["watch", "watches", "timepiece", "timepieces", "wristwatch"]):
        return (
            "WATCHES INTENT: User is asking for WATCHES in general. "
            "List all watch stores from the context that match the user's query. "
            "Include stores that sell luxury, sports, dress, or affordable watches."
        )
    
    return ""


def _detect_optics_intent(question: str) -> str:
    """
    Detect specific optics/eyewear intent from user question.
    Returns an optics hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "glasses" -> hint to only show eyewear stores
    - "sunglasses" -> hint to only show sunglasses stores
    - "prescription glasses" -> hint to only show prescription glasses stores
    """
    question_lower = question.lower()
    
    # Prescription Glasses intent
    if any(keyword in question_lower for keyword in ["prescription glasses", "prescription", "eye care", "contact lenses", "contact lens"]):
        return (
            "OPTICS INTENT: User is specifically asking for PRESCRIPTION GLASSES or EYE CARE. "
            "ONLY list stores whose context or tags explicitly mention prescription glasses, eye care services, or contact lenses. "
            "DO NOT list stores that only sell sunglasses unless they also mention prescription glasses or eye care."
        )
    
    # Sunglasses intent
    if any(keyword in question_lower for keyword in ["sunglasses", "sunglass", "sun glasses"]):
        return (
            "OPTICS INTENT: User is specifically asking for SUNGLASSES. "
            "ONLY list stores whose context or tags explicitly mention sunglasses or branded sunglasses. "
            "DO NOT list stores that only sell prescription glasses unless they also mention sunglasses."
        )
    
    # General Optics/Eyewear intent
    if any(keyword in question_lower for keyword in ["glasses", "eyewear", "optical", "optics", "vision"]):
        return (
            "OPTICS INTENT: User is asking for GLASSES/EYEWEAR in general. "
            "List all optical/eyewear stores from the context that match the user's query. "
            "Include stores that sell prescription glasses, sunglasses, or contact lenses."
        )
    
    return ""


def _detect_footwear_intent(question: str) -> str:
    """
    Detect specific footwear intent from user question.
    Returns a footwear hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "formal shoes" -> hint to only show formal shoe stores
    - "men formal shoes" -> hint to only show men's formal shoe stores (FOOTWEAR category only)
    - "casual shoes" -> hint to only show casual shoe stores
    - "sneakers" -> hint to only show sneaker stores
    - "khussa" -> hint to only show traditional footwear stores
    """
    question_lower = question.lower()
    
    # Check if question contains shoe/footwear keywords (must have at least one)
    has_shoe_keyword = any(keyword in question_lower for keyword in [
        "shoes", "shoe", "footwear", "sneakers", "sneaker", "sandals", "sandal", 
        "heels", "heel", "boots", "boot", "khussa", "khussas", "khusa"
    ])
    
    # Check for gender keywords
    has_men = any(keyword in question_lower for keyword in ["men", "men's", "mens", "male"])
    has_women = any(keyword in question_lower for keyword in ["women", "women's", "womens", "female"])
    has_kids = any(keyword in question_lower for keyword in ["kids", "kid's", "children", "children's"])
    
    # Formal Shoes intent (check first before general footwear)
    if any(keyword in question_lower for keyword in ["formal shoes", "formal shoe", "formal footwear", "dress shoes"]):
        gender_filter = ""
        if has_men:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Men' AND ('Formal Shoes' or 'Formal'). "
        elif has_women:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Women' AND ('Formal Shoes' or 'Formal'). "
        elif has_kids:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Kids' AND ('Formal Shoes' or 'Formal'). "
        else:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Formal Shoes' or 'Formal'. "
        
        return (
            f"FOOTWEAR INTENT: User is specifically asking for FORMAL SHOES. "
            f"{gender_filter}"
            f"CRITICAL: DO NOT list stores from 'Clothing Brands' category even if they have 'Formal' or 'Men' tags. "
            f"ONLY list stores from 'Footwear/Shoes' category. "
            f"DO NOT list stores that only sell casual shoes, sneakers, or sandals unless they also mention formal shoes."
        )
    
    # Casual Shoes intent
    if any(keyword in question_lower for keyword in ["casual shoes", "casual shoe", "casual footwear"]):
        gender_filter = ""
        if has_men:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Men' AND ('Casual Shoes' or 'Casual'). "
        elif has_women:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Women' AND ('Casual Shoes' or 'Casual'). "
        elif has_kids:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Kids' AND ('Casual Shoes' or 'Casual'). "
        else:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Casual Shoes' or 'Casual'. "
        
        return (
            f"FOOTWEAR INTENT: User is specifically asking for CASUAL SHOES. "
            f"{gender_filter}"
            f"CRITICAL: DO NOT list stores from 'Clothing Brands' category even if they have 'Casual' tags. "
            f"ONLY list stores from 'Footwear/Shoes' category."
        )
    
    # Sneakers intent
    if any(keyword in question_lower for keyword in ["sneakers", "sneaker", "sneaker shoes"]):
        gender_filter = ""
        if has_men:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Men' AND 'Sneakers'. "
        elif has_women:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Women' AND 'Sneakers'. "
        elif has_kids:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Kids' AND 'Sneakers'. "
        else:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Sneakers'. "
        
        return (
            f"FOOTWEAR INTENT: User is specifically asking for SNEAKERS. "
            f"{gender_filter}"
            f"CRITICAL: DO NOT list stores from 'Clothing Brands' category. "
            f"ONLY list stores from 'Footwear/Shoes' category."
        )
    
    # Traditional Footwear (Khussa) intent
    if any(keyword in question_lower for keyword in ["khussa", "khussas", "khusa", "traditional khussa", "handcrafted khussa", "ethnic sandals"]):
        return (
            "FOOTWEAR INTENT: User is specifically asking for TRADITIONAL FOOTWEAR (KHUSSA). "
            "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Traditional Footwear' or mention khussa/khussas. "
            "CRITICAL: DO NOT list stores from 'Clothing Brands' category. "
            "ONLY list stores from 'Footwear/Shoes' category."
        )
    
    # Athletic/Performance Footwear intent
    if any(keyword in question_lower for keyword in ["athletic footwear", "athletic shoes", "running shoes", "sports shoes", "performance footwear"]):
        return (
            "FOOTWEAR INTENT: User is specifically asking for ATHLETIC/PERFORMANCE FOOTWEAR. "
            "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Athletic Footwear' or 'Running Shoes'. "
            "CRITICAL: DO NOT list stores from 'Clothing Brands' or 'Sports & Activewear' categories. "
            "ONLY list stores from 'Footwear/Shoes' category."
        )
    
    # Heels intent
    if any(keyword in question_lower for keyword in ["heels", "heel", "high heels", "women's heels"]):
        return (
            "FOOTWEAR INTENT: User is specifically asking for HEELS. "
            "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Heels'. "
            "CRITICAL: DO NOT list stores from 'Clothing Brands' category. "
            "ONLY list stores from 'Footwear/Shoes' category."
        )
    
    # Sandals intent
    if any(keyword in question_lower for keyword in ["sandals", "sandal", "flip-flops", "flip flops"]):
        return (
            "FOOTWEAR INTENT: User is specifically asking for SANDALS. "
            "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Sandals' or 'Flip-Flops'. "
            "CRITICAL: DO NOT list stores from 'Clothing Brands' category. "
            "ONLY list stores from 'Footwear/Shoes' category."
        )
    
    # General Footwear intent (with shoe keyword check)
    if has_shoe_keyword:
        gender_filter = ""
        if has_men:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Men'. "
        elif has_women:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Women'. "
        elif has_kids:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes' AND tags include 'Kids'. "
        else:
            gender_filter = "ONLY list stores whose category is 'Footwear/Shoes'. "
        
        return (
            f"FOOTWEAR INTENT: User is asking for FOOTWEAR/SHOES. "
            f"{gender_filter}"
            f"CRITICAL: DO NOT list stores from 'Clothing Brands' category even if they have matching tags. "
            f"ONLY list stores from 'Footwear/Shoes' category. "
            f"Include stores that sell formal, casual, athletic, traditional, or any other type of footwear."
        )
    
    return ""


def _detect_clothing_intent(question: str) -> str:
    """
    Detect specific clothing intent from user question.
    Returns a clothing hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "men's clothing" -> hint to only show men's clothing stores
    - "women's formal wear" -> hint to only show women's formal wear stores
    - "ethnic wear" -> hint to only show ethnic/traditional clothing stores
    - "casual wear" -> hint to only show casual clothing stores
    """
    question_lower = question.lower()
    
    # Men's Clothing intent
    if any(keyword in question_lower for keyword in ["men's clothing", "menswear", "men's wear", "men's fashion", "men's apparel"]):
        return (
            "CLOTHING INTENT: User is specifically asking for MEN'S CLOTHING. "
            "ONLY list stores whose context or tags explicitly mention men's clothing, menswear, or men's fashion. "
            "DO NOT list stores that only sell women's or kids' clothing unless they also mention men's clothing."
        )
    
    # Women's Clothing intent
    if any(keyword in question_lower for keyword in ["women's clothing", "womenswear", "women's wear", "women's fashion", "women's apparel"]):
        return (
            "CLOTHING INTENT: User is specifically asking for WOMEN'S CLOTHING. "
            "ONLY list stores whose context or tags explicitly mention women's clothing, womenswear, or women's fashion. "
            "DO NOT list stores that only sell men's or kids' clothing unless they also mention women's clothing."
        )
    
    # Kids Clothing intent
    if any(keyword in question_lower for keyword in ["kids clothing", "kidswear", "kids wear", "children's clothing", "children's wear", "baby clothes"]):
        return (
            "CLOTHING INTENT: User is specifically asking for KIDS/CHILDREN'S CLOTHING. "
            "ONLY list stores whose context or tags explicitly mention kids clothing, children's clothing, or kidswear. "
            "DO NOT list stores that only sell men's or women's clothing unless they also mention kids/children's clothing."
        )
    
    # Formal Wear intent
    if any(keyword in question_lower for keyword in ["formal wear", "formal clothing", "formal", "suits", "dress shirts", "office wear"]):
        return (
            "CLOTHING INTENT: User is specifically asking for FORMAL WEAR. "
            "ONLY list stores whose context or tags explicitly mention formal wear, suits, dress shirts, or office wear. "
            "DO NOT list stores that only sell casual wear or streetwear unless they also mention formal wear."
        )
    
    # Casual Wear intent
    if any(keyword in question_lower for keyword in ["casual wear", "casual clothing", "casual", "everyday wear"]):
        return (
            "CLOTHING INTENT: User is specifically asking for CASUAL WEAR. "
            "ONLY list stores whose context or tags explicitly mention casual wear, casual clothing, or everyday wear. "
            "DO NOT list stores that only sell formal wear unless they also mention casual wear."
        )
    
    # Ethnic/Traditional Wear intent
    if any(keyword in question_lower for keyword in ["ethnic wear", "ethnic clothing", "traditional wear", "traditional clothing", "eastern wear", "kurtas", "unstitched", "shalwar kameez"]):
        return (
            "CLOTHING INTENT: User is specifically asking for ETHNIC/TRADITIONAL WEAR. "
            "ONLY list stores whose context or tags explicitly mention ethnic wear, traditional wear, eastern wear, kurtas, or unstitched fabrics. "
            "DO NOT list stores that only sell western wear unless they also mention ethnic/traditional wear."
        )
    
    # Western Wear intent
    if any(keyword in question_lower for keyword in ["western wear", "western clothing", "western", "denim", "jeans", "t-shirts"]):
        return (
            "CLOTHING INTENT: User is specifically asking for WESTERN WEAR. "
            "ONLY list stores whose context or tags explicitly mention western wear, western clothing, denim, jeans, or t-shirts. "
            "DO NOT list stores that only sell ethnic/traditional wear unless they also mention western wear."
        )
    
    # Fusion Wear intent
    if any(keyword in question_lower for keyword in ["fusion wear", "fusion clothing", "fusion"]):
        return (
            "CLOTHING INTENT: User is specifically asking for FUSION WEAR. "
            "ONLY list stores whose context or tags explicitly mention fusion wear or fusion clothing. "
            "DO NOT list stores that only sell pure ethnic or pure western wear unless they also mention fusion."
        )
    
    # Streetwear intent
    if any(keyword in question_lower for keyword in ["streetwear", "street wear", "urban wear", "urban clothing"]):
        return (
            "CLOTHING INTENT: User is specifically asking for STREETWEAR. "
            "ONLY list stores whose context or tags explicitly mention streetwear, street wear, or urban wear. "
            "DO NOT list stores that only sell formal or traditional wear unless they also mention streetwear."
        )
    
    # Bridal Wear intent
    if any(keyword in question_lower for keyword in ["bridal wear", "bridal clothing", "bridal", "bridal couture", "wedding wear"]):
        return (
            "CLOTHING INTENT: User is specifically asking for BRIDAL WEAR. "
            "ONLY list stores whose context or tags explicitly mention bridal wear, bridal couture, or wedding wear. "
            "DO NOT list stores that only sell casual or everyday wear unless they also mention bridal wear."
        )
    
    # Smart Casual intent
    if any(keyword in question_lower for keyword in ["smart casual", "smart-casual"]):
        return (
            "CLOTHING INTENT: User is specifically asking for SMART CASUAL WEAR. "
            "ONLY list stores whose context or tags explicitly mention smart casual or smart-casual wear. "
            "DO NOT list stores that only sell formal or pure casual wear unless they also mention smart casual."
        )
    
    # Modest Wear intent
    if any(keyword in question_lower for keyword in ["modest wear", "modest clothing", "abaya", "abayas"]):
        return (
            "CLOTHING INTENT: User is specifically asking for MODEST WEAR. "
            "ONLY list stores whose context or tags explicitly mention modest wear, abayas, or modest fashion. "
            "DO NOT list stores that only sell regular clothing unless they also mention modest wear."
        )
    
    # General Clothing intent
    if any(keyword in question_lower for keyword in ["clothing", "clothes", "apparel", "fashion", "garments"]):
        return (
            "CLOTHING INTENT: User is asking for CLOTHING in general. "
            "List all clothing stores from the context that match the user's query. "
            "Include stores that sell men's, women's, kids', formal, casual, ethnic, western, or any other type of clothing."
        )
    
    return ""


def _detect_services_intent(question: str) -> str:
    """
    Detect specific services intent from user question.
    Returns a services hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "grocery" -> hint to only show grocery stores
    - "bank" -> hint to only show banking services
    - "clinic" -> hint to only show clinics
    - "currency exchange" -> hint to only show currency exchange services
    """
    question_lower = question.lower()
    
    # Grocery intent
    if any(keyword in question_lower for keyword in ["grocery", "groceries", "fresh produce", "household essentials"]):
        return (
            "SERVICES INTENT: User is specifically asking for GROCERY stores. "
            "ONLY list stores whose context or tags explicitly mention grocery, groceries, fresh produce, or household essentials. "
            "DO NOT list stores from other service categories unless they also mention grocery."
        )
    
    # Banking intent
    if any(keyword in question_lower for keyword in ["bank", "banking", "atm", "islamic banking", "financial solutions", "loans"]):
        return (
            "SERVICES INTENT: User is specifically asking for BANKING services. "
            "ONLY list stores whose context or tags explicitly mention bank, banking, ATM, Islamic banking, or financial solutions. "
            "DO NOT list stores from other service categories unless they also mention banking."
        )
    
    # Clinic/Medical intent
    if any(keyword in question_lower for keyword in ["clinic", "medical center", "dental", "dermatology", "aesthetic", "cosmetic", "iv therapy"]):
        return (
            "SERVICES INTENT: User is specifically asking for CLINIC/MEDICAL services. "
            "ONLY list stores whose context or tags explicitly mention clinic, medical center, dental, dermatology, aesthetic, or IV therapy. "
            "DO NOT list stores from other service categories unless they also mention clinic/medical services."
        )
    
    # Currency Exchange intent
    if any(keyword in question_lower for keyword in ["currency exchange", "currency", "money transfer", "western union", "moneygram"]):
        return (
            "SERVICES INTENT: User is specifically asking for CURRENCY EXCHANGE services. "
            "ONLY list stores whose context or tags explicitly mention currency exchange, money transfer, Western Union, or MoneyGram. "
            "DO NOT list stores from other service categories unless they also mention currency exchange."
        )
    
    # Religious intent
    if any(keyword in question_lower for keyword in ["mosque", "prayer hall", "prayer", "religious"]):
        return (
            "SERVICES INTENT: User is specifically asking for RELIGIOUS services. "
            "ONLY list stores whose context or tags explicitly mention mosque, prayer hall, or religious services. "
            "DO NOT list stores from other service categories unless they also mention religious services."
        )
    
    # General Services intent
    if any(keyword in question_lower for keyword in ["services", "service"]):
        return (
            "SERVICES INTENT: User is asking for SERVICES in general. "
            "List all service stores from the context that match the user's query. "
            "Include stores that offer grocery, banking, clinic, currency exchange, or religious services."
        )
    
    return ""


def _detect_entertainment_intent(question: str) -> str:
    """
    Detect specific entertainment intent from user question.
    Returns an entertainment hint string for the prompt, or empty string if no specific intent.
    
    Examples:
    - "cinema" -> hint to only show cinema/movie theaters
    - "arcade" -> hint to only show arcade/gaming stores
    - "bowling" -> hint to only show bowling alleys
    - "family entertainment" -> hint to only show family entertainment venues
    """
    question_lower = question.lower()
    
    # Cinema intent
    if any(keyword in question_lower for keyword in ["cinema", "movie", "movies", "blockbusters", "movie theater", "movie theatre"]):
        return (
            "ENTERTAINMENT INTENT: User is specifically asking for CINEMA/MOVIES. "
            "ONLY list stores whose context or tags explicitly mention cinema, movies, or movie theaters. "
            "DO NOT list stores from other entertainment categories unless they also mention cinema/movies."
        )
    
    # Arcade & Gaming intent
    if any(keyword in question_lower for keyword in ["arcade", "arcade games", "gaming", "shooting game", "simulators", "vr", "virtual reality"]):
        return (
            "ENTERTAINMENT INTENT: User is specifically asking for ARCADE/GAMING. "
            "ONLY list stores whose context or tags explicitly mention arcade, arcade games, gaming, shooting games, simulators, or VR. "
            "DO NOT list stores from other entertainment categories unless they also mention arcade/gaming."
        )
    
    # Bowling intent
    if any(keyword in question_lower for keyword in ["bowling", "bowling alley", "bowling lanes"]):
        return (
            "ENTERTAINMENT INTENT: User is specifically asking for BOWLING. "
            "ONLY list stores whose context or tags explicitly mention bowling or bowling alley. "
            "DO NOT list stores from other entertainment categories unless they also mention bowling."
        )
    
    # Family Entertainment intent
    if any(keyword in question_lower for keyword in ["family entertainment", "rides", "attractions", "fun express", "train ride"]):
        return (
            "ENTERTAINMENT INTENT: User is specifically asking for FAMILY ENTERTAINMENT. "
            "ONLY list stores whose context or tags explicitly mention family entertainment, rides, attractions, or fun express. "
            "DO NOT list stores from other entertainment categories unless they also mention family entertainment."
        )
    
    # General Entertainment intent
    if any(keyword in question_lower for keyword in ["entertainment", "fun", "recreation"]):
        return (
            "ENTERTAINMENT INTENT: User is asking for ENTERTAINMENT in general. "
            "List all entertainment venues from the context that match the user's query. "
            "Include stores that offer cinema, arcade, bowling, family entertainment, or other recreational activities."
        )
    
    return ""


def _filter_docs_by_multiple_tags(docs: List[Any], question: str) -> List[Any]:
    """
    Filter documents to require at least 2 matching tags when query contains multiple keywords.
    Also ensures category matches the query intent.
    
    Args:
        docs: List of retrieved documents
        question: User's question
    
    Returns:
        Filtered list of documents
    """
    question_lower = question.lower()
    
    # Extract required tags from question
    required_tags = []
    required_category = None
    
    # Gender tags
    if any(k in question_lower for k in ["men", "men's", "mens", "male"]):
        required_tags.append("men")
    if any(k in question_lower for k in ["women", "women's", "womens", "female"]):
        required_tags.append("women")
    if any(k in question_lower for k in ["kids", "kid's", "children", "children's"]):
        required_tags.append("kids")
    
    # Footwear queries
    if any(k in question_lower for k in ["shoes", "shoe", "footwear"]):
        required_category = "Footwear/Shoes"
        if any(k in question_lower for k in ["formal shoes", "formal shoe", "formal footwear", "dress shoes"]):
            required_tags.append("formal")
        elif any(k in question_lower for k in ["casual shoes", "casual shoe", "casual footwear"]):
            required_tags.append("casual")
        elif any(k in question_lower for k in ["sneakers", "sneaker"]):
            required_tags.append("sneakers")
        else:
            # General footwear query - still require category match
            required_tags.append("footwear")  # This will match any footwear tag
    
    # Clothing queries (only if not footwear)
    elif any(k in question_lower for k in ["clothing", "clothes", "wear", "attire", "fashion"]):
        required_category = "Clothing Brands"
        if any(k in question_lower for k in ["formal wear", "formal", "suits", "dress shirts"]):
            required_tags.append("formal")
        elif any(k in question_lower for k in ["casual wear", "casual"]):
            required_tags.append("casual")
    
    # If we have multiple required tags, filter strictly
    if len(required_tags) >= 2 or required_category:
        filtered_docs = []
        for doc in docs:
            metadata = getattr(doc, "metadata", {})
            tags_str = metadata.get("tags", "")
            category = metadata.get("category", "")
            tags_list = [tag.strip().lower() for tag in tags_str.split(",") if tag.strip()] if tags_str else []
            
            # Category check - must match if specified
            if required_category and category != required_category:
                continue  # Skip stores that don't match required category
            
            # Count matching required tags
            matches = 0
            for required_tag in required_tags:
                # Check if required tag matches any tag in the document
                tag_matched = False
                for tag in tags_list:
                    # For "footwear" keyword, match any footwear-related tag
                    if required_tag == "footwear":
                        if any(ft in tag for ft in ["footwear", "shoes", "shoe", "sneaker", "heels", "sandals"]):
                            tag_matched = True
                            break
                    # For other tags, check for exact or partial match
                    elif required_tag.lower() in tag.lower() or tag.lower() in required_tag.lower():
                        tag_matched = True
                        break
                
                if tag_matched:
                    matches += 1
            
            # Require at least 2 matches if we have multiple required tags
            if len(required_tags) >= 2:
                if matches >= 2:
                    filtered_docs.append(doc)
            elif required_category:
                # If category is required, at least 1 tag match is needed
                if matches >= 1:
                    filtered_docs.append(doc)
            else:
                filtered_docs.append(doc)
        
        # Return filtered docs if we have matches, otherwise return original (to avoid empty results)
        return filtered_docs if filtered_docs else docs
    
    return docs  # Return original if not enough keywords to filter


def _format_docs(docs: List[Any]) -> str:
    """
    Join retrieved documents into a single context string.
    Also logs the retrieved chunks for debugging.
    """
    # Log retrieved chunks with metadata (visible in terminal and log file)
    rag_logger.info("\n" + "=" * 80)
    rag_logger.info(f"RETRIEVED CHUNKS (Total: {len(docs)}):")
    rag_logger.info("=" * 80)
    for i, doc in enumerate(docs, 1):
        page_content = getattr(doc, "page_content", "")
        metadata = getattr(doc, "metadata", {})
        rag_logger.info(f"\n--- Chunk {i}/{len(docs)} ---")
        rag_logger.info(f"Store Name: {metadata.get('store_name', 'N/A')}")
        rag_logger.info(f"Floor: {metadata.get('floor', 'N/A')}")
        rag_logger.info(f"Type: {metadata.get('type', 'N/A')}")
        rag_logger.info(f"Category: {metadata.get('category', 'N/A')}")
        rag_logger.info(f"Sub-Category: {metadata.get('sub_category', 'N/A')}")
        rag_logger.info(f"Tags: {metadata.get('tags', 'N/A')}")
        rag_logger.info(f"\nFull Content:\n{page_content}")
        rag_logger.info("-" * 80)
    rag_logger.info("=" * 80 + "\n")
    
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
You behave like a warm mall concierge — polite, light-hearted, conversational, and always helpful, while providing accurate mall information.

GOALS:

Answer mall-related questions correctly

Handle casual conversation naturally

Guide users toward shopping, dining, or services whenever possible

==================================================
ALWAYS AVAILABLE INFORMATION

MALL LOCATION:
If the user asks about Giga Mall’s location, address, directions, or map:
"You can find Giga Mall at: https://maps.app.goo.gl/2sDgo5JKupbKcbCQ6
"

MALL CONTACT:
Phone: (051) 8491040

==================================================
KNOWLEDGE BASE CONTEXT

{context}

Each context item contains:

Store name

Floor number

Store type (Outlet or Kiosk)

Short description

Context Helping Instructions:
We have 7 floors in the mall:

Basement 1

LG Floor

Mezzanine Floor

Ground Floor

1st Floor

2nd Floor

2A Floor

==================================================
CONVERSATION HISTORY

{chat_history_section}

Use history to understand follow-ups, emotional tone, and context for short replies.

==================================================
CURRENT USER QUESTION

{question}

==================================================
INTENT UNDERSTANDING RULES (CRITICAL)

INTENT TYPE A: FOLLOW-UP / CONTINUATION
User says: "any other?", "more?", "what else?", "others?"

Continue the LAST discussed topic using context

Prefer new options not already listed

INTENT TYPE B: SOCIAL / CASUAL CHAT
User says: "i love you", "haha", "nice", "cool", "will you marry me?"

Respond warmly and politely

Do NOT give mall phone fallback

Do NOT hallucinate personal relationships

Gently steer back to mall help

Example tone:
"That’s very sweet! 😊 I’m always here to help you enjoy Giga Mall. Want food, shopping, or something fun today?"

INTENT TYPE C: STORE VS MALL

Store names (e.g., J., Junaid Jamshed, Cheezious) refer to stores

Mall location rules apply ONLY to Giga Mall itself

==================================================
INTENT HINTS (IF PROVIDED)

{cuisine_hint}

{beauty_hint}

{sports_hint}

{electronics_hint}

{jewelry_hint}

{watches_hint}

{optics_hint}

{footwear_hint}

{clothing_hint}

{services_hint}

{entertainment_hint}

If an intent hint is provided above, you MUST strictly follow it.
Only list stores that match the specified intent type.
Do NOT include stores that don't match the intent.

CRITICAL TAG MATCHING REQUIREMENTS:
- When a user query contains multiple keywords (e.g., "men formal shoes"), stores MUST match AT LEAST 2 relevant tags from the query.
- For "men formal shoes": store must have BOTH "Men" AND ("Formal Shoes" or "Formal") tags, AND category must be "Footwear/Shoes".
- For "women casual clothing": store must have BOTH "Women" AND "Casual" tags, AND category must be "Clothing Brands".
- DO NOT list stores that match only 1 tag when the query contains multiple keywords.
- ALWAYS verify the store's category matches the query intent (e.g., footwear queries → Footwear/Shoes category only, clothing queries → Clothing Brands category only).
- If a store's category doesn't match the query intent, EXCLUDE it even if it has matching tags.
- Atleast provide 5 stores in the response if available, if more than 5 stores are available, provide all of them.
==================================================
RESPONSE RULES (APPLY IN ORDER)

RULE 1: MALL LOCATION

Only if the mall itself is mentioned

RULE 2: PRICES / DEALS / MENU

If pricing or deals are asked:
"For product information, deals, and pricing details, please visit the Giga Mall website or contact the store directly."

RULE 3: STORE / DINING INFO (FROM CONTEXT)

Use when store name, category, or related follow-up is mentioned

Food, shopping, kids, entertainment, fragrances, clothing, etc.

Follow-up intent is detected

RULE 4: STORE INFO (PRODUCT CATEGORIES)

Footwear Stores → footwear and accessories only

Clothing Stores → clothing and accessories only

Electronics Stores → electronics and accessories only

Homeware Stores → homeware and accessories only

Beauty Stores → beauty and accessories only

Food Stores → food and accessories only

Furniture Stores → furniture and accessories only

Other Stores → other products only

Category Restrictions (explicit)

No footwear store sells clothing or accessories

No clothing store sells footwear or accessories

No electronics store sells furniture or accessories

No homeware store sells electronics or accessories

No beauty store sells furniture or accessories

No food store sells beauty or accessories

No furniture store sells food or accessories

No other store sells food, beauty, or accessories

If multiple matching stores are found, prefer stores whose description contains "TOP PICK" and list them first. 

RULE 4a: SPECIAL STORE – KHAADI

Khaadi is a unique multi-category store

Offers:

Women’s clothing (unstiched and ready-to-wear)

Women’s fragrances

Khaadi Home (home care products)

When responding about Khaadi:

Include relevant category(s) based on user query

Follow the same “2–5 options, avoid repeats” rule

Format exactly as:

Khaadi - Floor X (Outlet): Women’s clothing, fragrances, or home care products

RULE 4b: RESPONSE FORMAT

Use ONLY provided context

Keep the response frontend friendly and engaging.

Avoid repeating stores already mentioned

Format exactly as:

Store Name - Floor X (Outlet/Kiosk): Short description

RULE 5: OUT OF DOMAIN (LAST RESORT)

Only if:

Not a follow-up

Not social chat

Not mall-related

Response:
"I’m unable to respond to your query. Please contact Giga Mall at (051) 8491040 for assistance."

RULE 6: ADULT CONTENT

Any adult-related question → "I cannot answer that question."

==================================================
STYLE & TONE RULES

Friendly, cheerful, human

Plain text only

Short, clear responses

Emojis allowed sparingly 😊🍔🛍️

Never sound robotic

==================================================
FINAL BEHAVIOR PRINCIPLE

Be helpful first, warm always, strict only when necessary. Use only the provided context and conversation history to answer.
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
        
        # Detect intent from question (cuisine, beauty, sports, electronics, jewelry, watches, optics, footwear, clothing, services, and entertainment)
        cuisine_hint = _detect_cuisine_intent(question)
        beauty_hint = _detect_beauty_intent(question)
        sports_hint = _detect_sports_intent(question)
        electronics_hint = _detect_electronics_intent(question)
        jewelry_hint = _detect_jewelry_intent(question)
        watches_hint = _detect_watches_intent(question)
        optics_hint = _detect_optics_intent(question)
        footwear_hint = _detect_footwear_intent(question)
        clothing_hint = _detect_clothing_intent(question)
        services_hint = _detect_services_intent(question)
        entertainment_hint = _detect_entertainment_intent(question)
        
        # Default hints if no intent detected
        default_cuisine_hint = "No specific cuisine intent detected. List all relevant restaurants from the context."
        default_beauty_hint = "No specific beauty intent detected. List all relevant beauty/fragrance stores from the context."
        default_sports_hint = "No specific sports intent detected. List all relevant sports/activewear stores from the context."
        default_electronics_hint = "No specific electronics intent detected. List all relevant electronics/tech stores from the context."
        default_jewelry_hint = "No specific jewelry intent detected. List all relevant jewelry stores from the context."
        default_watches_hint = "No specific watches intent detected. List all relevant watch stores from the context."
        default_optics_hint = "No specific optics intent detected. List all relevant eyewear/optical stores from the context."
        default_footwear_hint = "No specific footwear intent detected. List all relevant footwear/shoe stores from the context."
        default_clothing_hint = "No specific clothing intent detected. List all relevant clothing stores from the context."
        default_services_hint = "No specific services intent detected. List all relevant service stores from the context."
        default_entertainment_hint = "No specific entertainment intent detected. List all relevant entertainment venues from the context."
        
        # Format the final prompt for logging
        formatted_prompt = prompt_template.format(
            context=context,
            question=question,
            chat_history_section=chat_history_section,
            cuisine_hint=cuisine_hint if cuisine_hint else default_cuisine_hint,
            beauty_hint=beauty_hint if beauty_hint else default_beauty_hint,
            sports_hint=sports_hint if sports_hint else default_sports_hint,
            electronics_hint=electronics_hint if electronics_hint else default_electronics_hint,
            jewelry_hint=jewelry_hint if jewelry_hint else default_jewelry_hint,
            watches_hint=watches_hint if watches_hint else default_watches_hint,
            optics_hint=optics_hint if optics_hint else default_optics_hint,
            footwear_hint=footwear_hint if footwear_hint else default_footwear_hint,
            clothing_hint=clothing_hint if clothing_hint else default_clothing_hint,
            services_hint=services_hint if services_hint else default_services_hint,
            entertainment_hint=entertainment_hint if entertainment_hint else default_entertainment_hint,
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
            "cuisine_hint": cuisine_hint if cuisine_hint else default_cuisine_hint,
            "beauty_hint": beauty_hint if beauty_hint else default_beauty_hint,
            "sports_hint": sports_hint if sports_hint else default_sports_hint,
            "electronics_hint": electronics_hint if electronics_hint else default_electronics_hint,
            "jewelry_hint": jewelry_hint if jewelry_hint else default_jewelry_hint,
            "watches_hint": watches_hint if watches_hint else default_watches_hint,
            "optics_hint": optics_hint if optics_hint else default_optics_hint,
            "footwear_hint": footwear_hint if footwear_hint else default_footwear_hint,
            "clothing_hint": clothing_hint if clothing_hint else default_clothing_hint,
            "services_hint": services_hint if services_hint else default_services_hint,
            "entertainment_hint": entertainment_hint if entertainment_hint else default_entertainment_hint,
        }

    # Wrap format_inputs in RunnableLambda to make it compatible with LCEL
    format_inputs_runnable = RunnableLambda(format_inputs)
    
    # Create a filter function that has access to the question
    def filter_and_format(inputs: dict) -> dict:
        """Filter docs by multiple tags and format them."""
        question = inputs.get("question", "")
        docs = inputs.get("context_docs", [])
        
        # Filter docs if needed
        filtered_docs = _filter_docs_by_multiple_tags(docs, question)
        
        # Format filtered docs
        context = _format_docs(filtered_docs)
        
        return {
            "context": context,
            "question": question,
        }
    
    filter_and_format_runnable = RunnableLambda(filter_and_format)

    rag_chain = (
        {
            "context_docs": retriever,
            "question": RunnablePassthrough(),
        }
        | filter_and_format_runnable
        | format_inputs_runnable
        | prompt
        | llm
    )

    return rag_chain

