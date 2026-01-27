"""
RAG service for markdown documents.

This module:
- Parses the Giga Mall markdown into one Document per store
- Adds rich metadata (store_name, floor, category, etc.)
- Applies semantic chunking per store
- Builds a Chroma vector store with persistent storage
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def _normalize_text(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters for robust matching."""
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _keyword_matches(text: str, keyword: str) -> bool:
    """
    Check if keyword matches in text using whole-word matching for single words,
    or exact phrase matching for multi-word phrases.
    
    Examples:
    - "desi" will match "desi food" but NOT "designs"
    - "chinese food" will match "chinese food restaurant" but NOT "chinese sportswear"
    - "fast food" will match "fast food restaurant" but NOT "fast fashion"
    """
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    # For multi-word phrases, use simple substring matching
    if " " in keyword_lower or "-" in keyword_lower:
        return keyword_lower in text_lower
    
    # For single words, use word boundary matching to avoid false positives
    # e.g., "desi" should match "desi food" but NOT "designs"
    pattern = r'\b' + re.escape(keyword_lower) + r'\b'
    return bool(re.search(pattern, text_lower))


# ============================================================================
# TAG MAPPINGS BY CATEGORY
# ============================================================================

# Dine/Food tags - keyword -> tag name mapping
# Note: Only use specific food-related phrases, NOT standalone words like "chinese" or "desi"
# to avoid false positives (e.g., "chinese sportswear" or "designs")
DINE_TAGS = {
    # Cuisine types - use specific phrases only
    "chinese food": "Chinese food",
    "chinese cuisine": "Chinese food",
    "chinese dishes": "Chinese food",
    "chinese-style": "Chinese food",
    "chinese flavors": "Chinese food",
    "chinese-inspired": "Chinese food",
    "chinese restaurant": "Chinese food",
    "chinese mains": "Chinese food",
    "desi food": "Desi food",
    "desi cuisine": "Desi food",
    "desi dishes": "Desi food",
    "desi restaurant": "Desi food",
    "desi gravies": "Desi food",
    "desi street food": "Desi food",
    "desi snacks": "Desi food",
    "desi-inspired": "Desi food",
    "pakistani food": "Desi food",
    "pakistani cuisine": "Desi food",
    "pakistani dishes": "Desi food",
    # Arabic cuisine - specific phrases only
    "arabic food": "Arabic cuisine",
    "arabic cuisine": "Arabic cuisine",
    "arabic dishes": "Arabic cuisine",
    "arabic and desi-inspired": "Arabic cuisine",
    "arabic-inspired": "Arabic cuisine",
    "mandi": "Arabic cuisine",  # Mandi is Arabic rice dish
    
    # Food items
    "pizza": "Pizza",
    "pizzas": "Pizza",
    "burger": "Burgers",
    "burgers": "Burgers",
    "pasta": "Pasta",
    "lasagna": "Pasta",
    "spaghetti": "Pasta",
    "fries": "Fries",
    "french fries": "Fries",
    "chicken": "Chicken",
    "grilled chicken": "Chicken",
    "chicken wings": "Chicken",
    "kebab": "BBQ",
    "kebabs": "BBQ",
    "bbq": "BBQ",
    "grilled": "BBQ",
    "biryani": "Rice dishes",
    "rice dishes": "Rice dishes",
    "fried rice": "Rice dishes",
    
    # Fast food
    "fast food": "Fast food",
    "fast-food": "Fast food",
    "quick meals": "Fast food",
    "quick food": "Fast food",
    
    # Beverages
    "coffee": "Cafe",
    "cafe": "Cafe",
    "latte": "Cafe",
    "cappuccino": "Cafe",
    "espresso": "Cafe",
    "juice": "Juice",
    "juices": "Juice",
    "fresh juice": "Juice",
    "fruit juice": "Juice",
    "tea": "Tea",
    "chai": "Tea",
    "green tea": "Tea",
    "pink tea": "Tea",
    "bubble tea": "Bubble tea",
    "boba": "Bubble tea",
    "milk tea": "Bubble tea",
    
    # Desserts & Snacks
    "ice cream": "Ice cream",
    "ice-cream": "Ice cream",
    "frozen treats": "Ice cream",
    "dessert": "Desserts",
    "desserts": "Desserts",
    "sweet treats": "Desserts",
    "pastries": "Desserts",
    "candy": "Candies",
    "candies": "Candies",
    "chocolates": "Candies",
    "sweets": "Candies",
    "chaat": "Street food",
    "street food": "Street food",
    "snacks": "Street food",
    "snack": "Street food",
    
    # Noodles & Rice
    "noodles": "Noodles",
    "chow mein": "Noodles",
    "stir-fry": "Noodles",
    "wok": "Noodles",
}

# Clothing tags - keyword -> tag name mapping
CLOTHING_TAGS = {
    # Gender
    "men": "Men",
    "menswear": "Men",
    "men's": "Men",
    "women": "Women",
    "womenswear": "Women",
    "women's": "Women",
    "kids": "Kids",
    "children": "Kids",
    "kidswear": "Kids",
    "baby": "Kids",
    "infant": "Kids",
    "toddler": "Kids",
    "unisex": "Unisex",
    
    # Style
    "casual": "Casual",
    "casual wear": "Casual",
    "formal": "Formal",
    "formal wear": "Formal",
    "smart casual": "Smart Casual",
    "smart-casual": "Smart Casual",
    "semi-formal": "Semi-Formal",
    "semi formal": "Semi-Formal",
    "ethnic": "Ethnic",
    "eastern": "Ethnic",
    "eastern wear": "Ethnic",
    "traditional": "Traditional",
    "streetwear": "Streetwear",
    "street wear": "Streetwear",
    "party wear": "Party Wear",
    "party": "Party Wear",
    "bridal": "Bridal",
    "bridal couture": "Bridal",
    "western": "Western",
    "western wear": "Western",
    "fusion": "Fusion",
    "fusion wear": "Fusion",
    "contemporary": "Contemporary",
    "modest wear": "Modest Wear",
    "abaya": "Modest Wear",
    "sportswear": "Sportswear",
    "sports": "Sportswear",
    "activewear": "Activewear",
    "active wear": "Activewear",
    "innerwear": "Innerwear",
    "underwear": "Innerwear",
    "leisurewear": "Leisurewear",
    "loungewear": "Leisurewear",
    
    # Items - Men's
    "shirts": "Shirts",
    "shirt": "Shirts",
    "dress shirts": "Shirts",
    "dress shirt": "Shirts",
    "polos": "Polos",
    "polo": "Polos",
    "polo shirts": "Polos",
    "t-shirts": "T-Shirts",
    "t-shirt": "T-Shirts",
    "tees": "T-Shirts",
    "tee": "T-Shirts",
    "suits": "Suits",
    "suit": "Suits",
    "tailored suits": "Suits",
    "three-piece suits": "Suits",
    "trousers": "Trousers",
    "trouser": "Trousers",
    "pants": "Trousers",
    "chinos": "Chinos",
    "blazers": "Blazers",
    "blazer": "Blazers",
    "sherwani": "Ethnic",
    "prince coat": "Ethnic",
    "kameez": "Ethnic",
    "shalwar kameez": "Ethnic",
    
    # Items - Women's
    "kurtas": "Kurtas",
    "kurta": "Kurtas",
    "dresses": "Dresses",
    "dress": "Dresses",
    "tops": "Tops",
    "top": "Tops",
    "unstitched": "Unstitched",
    "unstitched fabrics": "Unstitched",
    "unstitched suits": "Unstitched",
    "ready-to-wear": "Ready-to-Wear",
    "ready to wear": "Ready-to-Wear",
    "pret": "Ready-to-Wear",
    "prêt": "Ready-to-Wear",
    "lawn": "Lawn",
    "chiffon": "Chiffon",
    "silk": "Silk",
    
    # Items - General
    "hoodies": "Hoodies",
    "hoodie": "Hoodies",
    "jackets": "Jackets",
    "jacket": "Jackets",
    "jeans": "Jeans",
    "denim": "Jeans",
    "shoes": "Shoes",
    "footwear": "Shoes",
    "accessories": "Accessories",
    "scarves": "Accessories",
    "bags": "Accessories",
    "handbags": "Accessories",
    
    # Price/Specialty
    "luxury": "Luxury",
    "premium": "Luxury",
    "high-end": "Luxury",
    "high-street": "High-Street",
    "high street": "High-Street",
    "budget": "Budget",
    "affordable": "Budget",
    "designer": "Designer",
    "multi-designer": "Designer",
    "fast fashion": "Fast Fashion",
    "fast-fashion": "Fast Fashion",
    "couture": "Couture",
    "bespoke": "Bespoke",
    "custom": "Custom",
    "tailored": "Tailored",
}

# Beauty & Fragrances tags - keyword -> tag name mapping
BEAUTY_TAGS = {
    # Fragrances
    "perfume": "Fragrances",
    "perfumes": "Fragrances",
    "fragrance": "Fragrances",
    "fragrances": "Fragrances",
    "attar": "Fragrances",
    "attars": "Fragrances",
    "cologne": "Fragrances",
    "colognes": "Fragrances",
    "scent": "Fragrances",
    "scents": "Fragrances",
    "perfume oils": "Fragrances",
    "perfume oil": "Fragrances",
    "concentrated perfume": "Fragrances",
    "body spray": "Fragrances",
    "body sprays": "Fragrances",
    "aromatic oils": "Fragrances",
    
    # Cosmetics/Makeup
    "makeup": "Cosmetics",
    "cosmetics": "Cosmetics",
    "foundation": "Cosmetics",
    "foundations": "Cosmetics",
    "lipstick": "Cosmetics",
    "lipsticks": "Cosmetics",
    "mascara": "Cosmetics",
    "mascaras": "Cosmetics",
    "nail polish": "Cosmetics",
    "nail polishes": "Cosmetics",
    "lip liner": "Cosmetics",
    "lip liners": "Cosmetics",
    "serum": "Cosmetics",
    "serums": "Cosmetics",
    "makeup essentials": "Cosmetics",
    
    # Skincare
    "skincare": "Skincare",
    "skin care": "Skincare",
    "beauty": "Skincare",
    "beauty products": "Skincare",
    "herbal": "Skincare",
    "natural": "Skincare",
    "organic": "Skincare",
    "haircare": "Skincare",
    "hair care": "Skincare",
    "body butters": "Skincare",
    "body butter": "Skincare",
    "skin treatments": "Skincare",
    "skin treatment": "Skincare",
    "personal care": "Skincare",
    "wellness": "Skincare",
    "essential oils": "Skincare",
    "essential oil": "Skincare",
    
    # Men's Grooming
    "men's grooming": "Men's Grooming",
    "mens grooming": "Men's Grooming",
    "grooming": "Men's Grooming",
    "beard": "Men's Grooming",
    "beard oil": "Men's Grooming",
    "beard oils": "Men's Grooming",
    "men's": "Men's Grooming",
    "mens": "Men's Grooming",
    
    # Fragrance Types
    "oriental": "Oriental Fragrances",
    "oriental scents": "Oriental Fragrances",
    "oud": "Oriental Fragrances",
    "ouds": "Oriental Fragrances",
    "musk": "Oriental Fragrances",
    "musks": "Oriental Fragrances",
    "traditional arabic": "Oriental Fragrances",
    "arabic attars": "Oriental Fragrances",
    "western": "Western Fragrances",
    "western-style": "Western Fragrances",
    "western scents": "Western Fragrances",
    "designer perfumes": "Designer Fragrances",
    "designer perfume": "Designer Fragrances",
    
    # Price/Specialty
    "luxury": "Luxury",
    "premium": "Luxury",
    "high-end": "Luxury",
    "budget": "Budget",
    "affordable": "Budget",
    "designer": "Designer",
    "cruelty-free": "Ethical",
    "sustainable": "Ethical",
    "ethical": "Ethical",
}

# Sports & Activewear tags - keyword -> tag name mapping
SPORTS_TAGS = {
    # Sportswear/Activewear
    "sportswear": "Sportswear",
    "sports wear": "Sportswear",
    "activewear": "Activewear",
    "active wear": "Activewear",
    "athletic apparel": "Athletic Apparel",
    "athletic wear": "Athletic Apparel",
    "gym gear": "Gym Gear",
    "gym wear": "Gym Gear",
    "fitness wear": "Fitness Wear",
    "performance apparel": "Performance Apparel",
    
    # Footwear
    "athletic footwear": "Athletic Footwear",
    "sports shoes": "Athletic Footwear",
    "sneakers": "Athletic Footwear",
    "sports footwear": "Athletic Footwear",
    "running shoes": "Running Shoes",
    "football boots": "Football Boots",
    "soccer cleats": "Football Boots",
    
    # Sports Equipment
    "sports equipment": "Sports Equipment",
    "sports gear": "Sports Equipment",
    "football kits": "Football Kits",
    "football kit": "Football Kits",
    "jerseys": "Jerseys",
    "jersey": "Jerseys",
    "club jerseys": "Jerseys",
    "tracksuits": "Tracksuits",
    "tracksuit": "Tracksuits",
    
    # Sports Types
    "football": "Football",
    "soccer": "Football",
    "cricket": "Cricket",
    "tennis": "Tennis",
    "basketball": "Basketball",
    "running": "Running",
    "gym": "Gym",
    "fitness": "Fitness",
    "training": "Training",
    
    # Specialty
    "professional sports": "Professional Sports",
    "professional gear": "Professional Sports",
    "performance": "Performance",
    "athletic": "Athletic",
    
    # Accessories
    "sports accessories": "Sports Accessories",
    "sports bags": "Sports Accessories",
    "fashion accessories": "Accessories",
    
    # Price/Specialty
    "luxury": "Luxury",
    "premium": "Luxury",
    "high-end": "Luxury",
    "budget": "Budget",
    "affordable": "Budget",
    "designer": "Designer",
}

# Electronics tags - keyword -> tag name mapping
ELECTRONICS_TAGS = {
    # Mobile/Smartphones
    "mobile": "Mobile",
    "mobiles": "Mobile",
    "smartphone": "Smartphones",
    "smartphones": "Smartphones",
    "phone": "Smartphones",
    "phones": "Smartphones",
    
    # Mobile Accessories
    "mobile accessories": "Mobile Accessories",
    "phone cases": "Mobile Accessories",
    "phone case": "Mobile Accessories",
    "chargers": "Mobile Accessories",
    "charger": "Mobile Accessories",
    "cables": "Mobile Accessories",
    "cable": "Mobile Accessories",
    "headphones": "Mobile Accessories",
    "headphone": "Mobile Accessories",
    "protective cases": "Mobile Accessories",
    "protective case": "Mobile Accessories",
    "high-speed chargers": "Mobile Accessories",
    "premium audio gear": "Mobile Accessories",
    "audio gear": "Mobile Accessories",
    
    # Electronics/Gadgets
    "electronics": "Electronics",
    "electronic": "Electronics",
    "gadgets": "Gadgets",
    "gadget": "Gadgets",
    "tech gadgets": "Gadgets",
    "electronic solutions": "Electronics",
    
    # Gaming
    "gaming": "Gaming",
    "gaming consoles": "Gaming",
    "gaming console": "Gaming",
    "gaming titles": "Gaming",
    "gaming accessories": "Gaming",
    "gaming gear": "Gaming",
    "playstation": "Gaming",
    "xbox": "Gaming",
    "nintendo": "Gaming",
    "video game": "Gaming",
    "video games": "Gaming",
    
    # Smart Home
    "smart home": "Smart Home",
    "smart home devices": "Smart Home",
    "air purifiers": "Smart Home",
    "air purifier": "Smart Home",
    "robot vacuums": "Smart Home",
    "robot vacuum": "Smart Home",
    
    # Audio
    "audio": "Audio",
    "audio gear": "Audio",
    "premium audio": "Audio",
    
    # Fitness Tech
    "fitness trackers": "Fitness Tech",
    "fitness tracker": "Fitness Tech",
    
    # Brands (if mentioned)
    "xiaomi": "Xiaomi",
    "mi": "Xiaomi",
    
    # Price/Specialty
    "luxury": "Luxury",
    "premium": "Luxury",
    "high-end": "Luxury",
    "budget": "Budget",
    "affordable": "Budget",
    "latest": "Latest Tech",
}

# Jewelry tags - keyword -> tag name mapping
JEWELRY_TAGS = {
    # Jewelry Types
    "jewelry": "Jewelry",
    "jewellery": "Jewelry",
    "fine jewelry": "Fine Jewelry",
    "fine jewellery": "Fine Jewelry",
    
    # Gold Jewelry
    "gold jewelry": "Gold Jewelry",
    "gold jewellery": "Gold Jewelry",
    "gold": "Gold Jewelry",
    "22k gold": "Gold Jewelry",
    "18kt gold": "Gold Jewelry",
    "18k gold": "Gold Jewelry",
    "traditional gold": "Gold Jewelry",
    "classic gold": "Gold Jewelry",
    
    # Diamond Jewelry
    "diamond jewelry": "Diamond Jewelry",
    "diamond jewellery": "Diamond Jewelry",
    "diamond": "Diamond Jewelry",
    "diamonds": "Diamond Jewelry",
    "diamond rings": "Diamond Jewelry",
    "diamond-studded": "Diamond Jewelry",
    "luxury diamond": "Diamond Jewelry",
    
    # Gemstone Jewelry
    "gemstone jewelry": "Gemstone Jewelry",
    "gemstone jewellery": "Gemstone Jewelry",
    "gemstone": "Gemstone Jewelry",
    "gemstones": "Gemstone Jewelry",
    "precious gemstone": "Gemstone Jewelry",
    "precious gemstones": "Gemstone Jewelry",
    "semi-precious": "Gemstone Jewelry",
    
    # Silver Jewelry
    "silver jewelry": "Silver Jewelry",
    "silver jewellery": "Silver Jewelry",
    "silver": "Silver Jewelry",
    "silver ornaments": "Silver Jewelry",
    "silver pieces": "Silver Jewelry",
    "gold-plated": "Silver Jewelry",
    
    # Bridal Jewelry
    "bridal jewelry": "Bridal Jewelry",
    "bridal jewellery": "Bridal Jewelry",
    "bridal sets": "Bridal Jewelry",
    "bridal set": "Bridal Jewelry",
    "bridal wear": "Bridal Jewelry",
    "bridal accessories": "Bridal Jewelry",
    "engagement rings": "Bridal Jewelry",
    "engagement ring": "Bridal Jewelry",
    
    # Traditional Jewelry
    "traditional jewelry": "Traditional Jewelry",
    "traditional jewellery": "Traditional Jewelry",
    "traditional craftsmanship": "Traditional Jewelry",
    "traditional ornaments": "Traditional Jewelry",
    "kundan": "Traditional Jewelry",
    "kundan work": "Traditional Jewelry",
    "handcrafted": "Traditional Jewelry",
    "heritage": "Traditional Jewelry",
    
    # Contemporary/Modern Jewelry
    "contemporary jewelry": "Contemporary Jewelry",
    "contemporary jewellery": "Contemporary Jewelry",
    "modern designs": "Contemporary Jewelry",
    "modern aesthetic": "Contemporary Jewelry",
    "high-fashion": "Contemporary Jewelry",
    
    # Crystal Jewelry
    "crystal jewelry": "Crystal Jewelry",
    "crystal jewellery": "Crystal Jewelry",
    "crystal": "Crystal Jewelry",
    "crystal creations": "Crystal Jewelry",
    "precision-cut crystal": "Crystal Jewelry",
    
    # Jewelry Items
    "rings": "Rings",
    "ring": "Rings",
    "bangles": "Bangles",
    "bangle": "Bangles",
    "necklaces": "Necklaces",
    "necklace": "Necklaces",
    "earrings": "Earrings",
    "earring": "Earrings",
    "watches": "Watches",
    "watch": "Watches",
    "figurines": "Figurines",
    "figurine": "Figurines",
    
    # Custom/Specialty
    "custom-crafted": "Custom Jewelry",
    "custom jewelry": "Custom Jewelry",
    "personalized": "Custom Jewelry",
    "lifetime warranty": "Custom Jewelry",
    
    # Price/Specialty
    "luxury": "Luxury",
    "premium": "Luxury",
    "high-end": "Luxury",
    "budget": "Budget",
    "affordable": "Budget",
    "reputed": "Reputed",
    "trusted": "Reputed",
}

# Watches tags - keyword -> tag name mapping
WATCHES_TAGS = {
    # Watches General
    "watch": "Watches",
    "watches": "Watches",
    "timepiece": "Watches",
    "timepieces": "Watches",
    "wristwatch": "Watches",
    "wristwatches": "Watches",
    
    # Watch Types
    "dress watch": "Dress Watches",
    "dress watches": "Dress Watches",
    "formal watch": "Dress Watches",
    "formal watches": "Dress Watches",
    "elegant watch": "Dress Watches",
    "sports watch": "Sports Watches",
    "sports watches": "Sports Watches",
    "chronograph": "Sports Watches",
    "chronographs": "Sports Watches",
    "sports-inspired": "Sports Watches",
    
    # Watch Materials
    "stainless steel": "Stainless Steel",
    "leather strap": "Leather Strap",
    "leather straps": "Leather Strap",
    
    # Gender
    "men": "Men",
    "men's": "Men",
    "women": "Women",
    "women's": "Women",
    
    # Price/Specialty
    "luxury": "Luxury",
    "affordable luxury": "Affordable Luxury",
    "premium": "Luxury",
    "high-end": "Luxury",
    "budget": "Budget",
    "affordable": "Budget",
    "original": "Original",
    "branded": "Branded",
    "100% original": "Original",
}

# Optics (Glasses & Eye Care) tags - keyword -> tag name mapping
OPTICS_TAGS = {
    # Glasses/Eyewear
    "glasses": "Glasses",
    "eyewear": "Eyewear",
    "prescription glasses": "Prescription Glasses",
    "prescription": "Prescription Glasses",
    "sunglasses": "Sunglasses",
    "contact lenses": "Contact Lenses",
    "contact lens": "Contact Lenses",
    "eye care": "Eye Care",
    "eye care services": "Eye Care",
    "optical": "Optical",
    "optics": "Optical",
    "vision": "Eye Care",
    "professional eye care": "Eye Care",
}

# Footwear/Shoes tags - keyword -> tag name mapping
FOOTWEAR_TAGS = {
    # Footwear General
    "footwear": "Footwear",
    "shoes": "Shoes",
    "shoe": "Shoes",
    
    # Shoe Types
    "formal shoes": "Formal Shoes",
    "formal shoe": "Formal Shoes",
    "formal footwear": "Formal Shoes",
    "casual shoes": "Casual Shoes",
    "casual shoe": "Casual Shoes",
    "casual footwear": "Casual Shoes",
    "sneakers": "Sneakers",
    "sneaker": "Sneakers",
    "sandals": "Sandals",
    "sandal": "Sandals",
    "heels": "Heels",
    "heel": "Heels",
    "high heels": "Heels",
    "boots": "Boots",
    "boot": "Boots",
    "flip-flops": "Flip-Flops",
    "flip flops": "Flip-Flops",
    "clogs": "Clogs",
    "wedges": "Wedges",
    "wedge": "Wedges",
    "joggers": "Joggers",
    "jogger": "Joggers",
    
    # Traditional Footwear
    "khussa": "Traditional Footwear",
    "khussas": "Traditional Footwear",
    "khusa": "Traditional Footwear",
    "traditional khussa": "Traditional Footwear",
    "handcrafted khussa": "Traditional Footwear",
    "ethnic sandals": "Traditional Footwear",
    "hand-stitched": "Traditional Footwear",
    "handcrafted": "Traditional Footwear",
    
    # Athletic/Performance
    "athletic footwear": "Athletic Footwear",
    "athletic shoes": "Athletic Footwear",
    "performance footwear": "Athletic Footwear",
    "running shoes": "Running Shoes",
    "sports shoes": "Athletic Footwear",
    "high-performance": "Athletic Footwear",
    "memory foam": "Comfort Technology",
    "comfort technology": "Comfort Technology",
    
    # Materials
    "leather shoes": "Leather Shoes",
    "leather footwear": "Leather Shoes",
    "premium leather": "Leather Shoes",
    "genuine leather": "Leather Shoes",
    
    # Gender
    "men": "Men",
    "men's": "Men",
    "women": "Women",
    "women's": "Women",
    "kids": "Kids",
    "children": "Kids",
    "children's": "Kids",
    
    # Style
    "elegant": "Elegant",
    "stylish": "Stylish",
    "trendy": "Trendy",
    "fashionable": "Fashionable",
    "designer shoes": "Designer",
    "designer footwear": "Designer",
    
    # Price/Specialty
    "luxury": "Luxury",
    "premium": "Luxury",
    "high-end": "Luxury",
    "budget": "Budget",
    "affordable": "Budget",
    "durable": "Durable",
    "comfortable": "Comfortable",
    "quality": "Quality",
}

# Services tags - keyword -> tag name mapping
SERVICES_TAGS = {
    # Grocery
    "grocery": "Grocery",
    "groceries": "Grocery",
    "fresh produce": "Grocery",
    "household essentials": "Grocery",
    "everyday necessities": "Grocery",
    
    # Beauty/Care Products
    "beauty products": "Beauty Products",
    "care products": "Beauty Products",
    "health and beauty": "Beauty Products",
    "wellness essentials": "Beauty Products",
    "personal care": "Beauty Products",
    
    # Banking
    "bank": "Banking",
    "banking": "Banking",
    "banking services": "Banking",
    "islamic banking": "Banking",
    "sharia-compliant": "Banking",
    "financial solutions": "Banking",
    "atm": "Banking",
    "atm access": "Banking",
    "personal banking": "Banking",
    "business banking": "Banking",
    "loans": "Banking",
    "account": "Banking",
    
    # Clinic/Medical
    "clinic": "Clinic",
    "medical center": "Clinic",
    "dental clinic": "Dental",
    "dental care": "Dental",
    "dental implants": "Dental",
    "dentures": "Dental",
    "dental cleaning": "Dental",
    "dental whitening": "Dental",
    "braces": "Dental",
    "dermatologist": "Dermatology",
    "dermatology": "Dermatology",
    "skincare clinic": "Dermatology",
    "aesthetic": "Aesthetics",
    "aesthetics": "Aesthetics",
    "cosmetic": "Aesthetics",
    "laser hair removal": "Aesthetics",
    "hair transplant": "Aesthetics",
    "botox": "Aesthetics",
    "fillers": "Aesthetics",
    "iv therapy": "IV Therapy",
    "iv therapies": "IV Therapy",
    "whitening injections": "IV Therapy",
    "glutathione drips": "IV Therapy",
    "fat loss drips": "IV Therapy",
    
    # Currency Exchange
    "currency exchange": "Currency Exchange",
    "currency": "Currency Exchange",
    "money transfer": "Currency Exchange",
    "western union": "Currency Exchange",
    "moneygram": "Currency Exchange",
    "ria money transfer": "Currency Exchange",
    
    # Religious
    "mosque": "Religious",
    "prayer hall": "Religious",
    "prayer": "Religious",
    "religious": "Religious",
}

# Entertainment tags - keyword -> tag name mapping
ENTERTAINMENT_TAGS = {
    # Cinema
    "cinema": "Cinema",
    "movie": "Cinema",
    "movies": "Cinema",
    "blockbusters": "Cinema",
    "screens": "Cinema",
    "movie experience": "Cinema",
    
    # Bowling
    "bowling": "Bowling",
    "bowling alley": "Bowling",
    "bowling lanes": "Bowling",
    
    # Arcade & Gaming
    "arcade": "Arcade",
    "arcade games": "Arcade",
    "gaming": "Gaming",
    "games": "Gaming",
    "shooting game": "Gaming",
    "simulators": "Gaming",
    "interactive experiences": "Gaming",
    "vr": "VR",
    "virtual reality": "VR",
    "vr experiences": "VR",
    
    # Family Entertainment
    "family entertainment": "Family Entertainment",
    "rides": "Family Entertainment",
    "attractions": "Family Entertainment",
    "train ride": "Family Entertainment",
    "fun express": "Family Entertainment",
    
    # Recreation
    "recreation": "Recreation",
    "sports and recreation": "Recreation",
    "entertainment": "Entertainment",
    "fun": "Entertainment",
}

# Universal tags (apply to all categories)
UNIVERSAL_TAGS = {
    "top pick": "Top Pick",
    "top pick –": "Top Pick",
    "recommended": "Top Pick",
}


def _parse_store_documents(markdown_text: str) -> List[Document]:
    """
    Parse the mall markdown into one Document per store with rich metadata.

    Heuristics:
    - Track current top-level group (# ...), e.g. "Stores/Shops at Giga Mall",
      "Dine/ Food Court / Meal options", "Services in Giga Mall", etc.
    - Track current category/sub-category (## ...), e.g. "Clothing Brands",
      "Fast Food", "Restaurant", "Cafe", etc.
    - Each line starting with "Store name:" starts a new store block.
    - All following non-empty lines until next "Store name:" or header are
      attached as description.
    """
    lines = markdown_text.splitlines()

    current_group: Optional[str] = None
    current_category: Optional[str] = None

    docs: List[Document] = []

    current_store_lines: List[str] = []
    current_store_metadata: Dict[str, Any] = {}

    def flush_current_store():
        if not current_store_lines:
            return
        text = "\n".join(current_store_lines).strip()
        if not text:
            return
        # Add normalized name for robust matching
        store_name = current_store_metadata.get("store_name")
        if store_name:
            current_store_metadata["normalized_store_name"] = _normalize_text(
                str(store_name)
            )
        
        # Extract tags based on category and description
        desc_lower = text.lower()
        keywords = []
        
        # Determine category type
        is_food_store = (
            current_group and "dine" in current_group.lower()
        ) or (
            current_category and any(word in current_category.lower() for word in [
                "food", "dine", "restaurant", "cafe", "snack", "juice", "ice cream", 
                "dessert", "burger", "chinese", "desi", "fast food", "street food"
            ])
        )
        
        is_clothing_store = (
            current_group and "stores" in current_group.lower() and "clothing" in current_group.lower()
        ) or (
            current_category and any(word in current_category.lower() for word in [
                "clothing", "wear", "fashion", "apparel", "menswear", "womenswear", "kids clothing"
            ])
        )
        
        is_beauty_store = (
            current_category and any(word in current_category.lower() for word in [
                "beauty", "fragrance", "fragrances", "cosmetics", "perfume", "skincare", "grooming"
            ])
        )
        
        is_sports_store = (
            current_category and any(word in current_category.lower() for word in [
                "sports", "sport", "activewear", "active wear", "athletic", "fitness", "gym"
            ])
        )
        
        is_electronics_store = (
            current_category and any(word in current_category.lower() for word in [
                "electronics", "electronic", "mobile", "gadgets", "gaming", "tech", "technology"
            ])
        )
        
        is_jewelry_store = (
            current_category and any(word in current_category.lower() for word in [
                "jewelry", "jewellery", "jeweler", "jeweller", "jewels"
            ])
        )
        
        is_watches_store = (
            current_category and any(word in current_category.lower() for word in [
                "watches", "watch", "timepiece", "timepieces"
            ])
        )
        
        is_optics_store = (
            current_category and any(word in current_category.lower() for word in [
                "optics", "optical", "glasses", "eyewear", "eye care", "sunglasses", "contact lenses"
            ])
        )
        
        is_footwear_store = (
            current_category and any(word in current_category.lower() for word in [
                "footwear", "shoes", "shoe"
            ])
        )
        
        is_services_store = (
            current_group and "services" in current_group.lower()
        ) or (
            current_category and any(word in current_category.lower() for word in [
                "grocery", "bank", "banking", "clinic", "currency exchange", "religious", "beauty", "care products"
            ])
        )
        
        is_entertainment_store = (
            current_group and "entertainment" in current_group.lower()
        ) or (
            current_category and any(word in current_category.lower() for word in [
                "cinema", "entertainment", "fun", "arcade", "gaming", "bowling", "recreation", "family entertainment"
            ])
        )
        
        # Extract tags using mappings with whole-word matching
        if is_food_store:
            # Check description against DINE_TAGS mapping
            for keyword, tag in DINE_TAGS.items():
                if _keyword_matches(desc_lower, keyword):
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        elif is_beauty_store:
            # Check description against BEAUTY_TAGS mapping
            for keyword, tag in BEAUTY_TAGS.items():
                if _keyword_matches(desc_lower, keyword):
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        elif is_sports_store:
            # Check description against SPORTS_TAGS mapping
            for keyword, tag in SPORTS_TAGS.items():
                if _keyword_matches(desc_lower, keyword):
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        elif is_electronics_store:
            # Check description against ELECTRONICS_TAGS mapping
            for keyword, tag in ELECTRONICS_TAGS.items():
                if _keyword_matches(desc_lower, keyword):
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        elif is_jewelry_store:
            # Check description against JEWELRY_TAGS mapping
            for keyword, tag in JEWELRY_TAGS.items():
                if _keyword_matches(desc_lower, keyword):
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        elif is_watches_store:
            # Check description against WATCHES_TAGS mapping
            for keyword, tag in WATCHES_TAGS.items():
                if _keyword_matches(desc_lower, keyword):
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        elif is_optics_store:
            # Check description against OPTICS_TAGS mapping
            for keyword, tag in OPTICS_TAGS.items():
                if _keyword_matches(desc_lower, keyword):
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        elif is_footwear_store:
            # Check description against FOOTWEAR_TAGS mapping
            for keyword, tag in FOOTWEAR_TAGS.items():
                if _keyword_matches(desc_lower, keyword):
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        elif is_services_store:
            # Check description against SERVICES_TAGS mapping
            for keyword, tag in SERVICES_TAGS.items():
                if _keyword_matches(desc_lower, keyword):
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        elif is_entertainment_store:
            # Check description against ENTERTAINMENT_TAGS mapping
            for keyword, tag in ENTERTAINMENT_TAGS.items():
                if _keyword_matches(desc_lower, keyword):
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        elif is_clothing_store:
            # Handle gender tags first (check unisex before individual genders)
            # Use whole-word matching for gender keywords
            has_men = _keyword_matches(desc_lower, "men") or "menswear" in desc_lower or "men's" in desc_lower
            has_women = _keyword_matches(desc_lower, "women") or "womenswear" in desc_lower or "women's" in desc_lower
            has_unisex = _keyword_matches(desc_lower, "unisex")
            
            if (has_men and has_women) or has_unisex:
                keywords.append("Unisex")
            elif has_men and not has_women:
                keywords.append("Men")
            elif has_women and not has_men:
                keywords.append("Women")
            
            # Check description against CLOTHING_TAGS mapping (skip gender tags already handled)
            gender_tags = {"Men", "Women", "Unisex"}
            for keyword, tag in CLOTHING_TAGS.items():
                if _keyword_matches(desc_lower, keyword) and tag not in gender_tags:
                    if tag not in keywords:  # Avoid duplicates
                        keywords.append(tag)
        
        # Universal tags (apply to all stores)
        for keyword, tag in UNIVERSAL_TAGS.items():
            if _keyword_matches(desc_lower, keyword):
                if tag not in keywords:  # Avoid duplicates
                    keywords.append(tag)
        
        # Convert tags list to comma-separated string for ChromaDB compatibility
        # ChromaDB only accepts str, int, float, bool, or None in metadata
        current_store_metadata["tags"] = ", ".join(keywords) if keywords else ""

        docs.append(
            Document(
                page_content=text,
                metadata=current_store_metadata.copy(),
            )
        )

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            # Blank line – just append to current store text if any
            if current_store_lines:
                current_store_lines.append("")
            continue

        # Top-level group header, e.g. "# Dine/ Food Court / Meal options"
        if line.startswith("# "):
            # Before switching group, flush any open store
            flush_current_store()
            current_store_lines = []
            current_store_metadata = {}

            current_group = line[2:].strip()
            # Reset category when group changes
            current_category = None
            continue

        # Category / sub-category header, e.g. "## Clothing Brands", "## Fast Food"
        if line.startswith("## "):
            # Flush any open store before changing category
            flush_current_store()
            current_store_lines = []
            current_store_metadata = {}

            current_category = line[3:].strip()
            continue

        # Store entry
        if line.startswith("Store name:"):
            # Flush previous store (if any)
            flush_current_store()
            current_store_lines = []
            current_store_metadata = {}

            # Basic metadata from context
            current_store_metadata["group"] = current_group

            # For Dine group, treat group as category and current_category as sub_category
            if current_group and "dine" in current_group.lower():
                current_store_metadata["category"] = current_group
                current_store_metadata["sub_category"] = current_category
            else:
                current_store_metadata["category"] = current_category
                current_store_metadata["sub_category"] = None

            # Parse inline metadata from "Store name: ... , Floor: X, Outlet, ..."
            store_line = line[len("Store name:") :].strip()
            parts = [p.strip() for p in store_line.split(",") if p.strip()]

            store_name: Optional[str] = None
            floor: Optional[int] = None
            store_type: Optional[str] = None

            if parts:
                store_name = parts[0]

            for part in parts[1:]:
                # Floor info
                if part.lower().startswith("floor"):
                    # e.g. "Floor: 4" or "Floor: -0"
                    match = re.search(r"(-?\d+)", part)
                    if match:
                        try:
                            floor = int(match.group(1))
                        except ValueError:
                            floor = None
                # Simple store type heuristic: single word like "Outlet", "Kiosk"
                elif part.lower() in {"outlet", "kiosk"}:
                    store_type = part

            if store_name:
                current_store_metadata["store_name"] = store_name
            if floor is not None:
                current_store_metadata["floor"] = floor
            if store_type:
                current_store_metadata["type"] = store_type

            # Start store text with this line (keep original for LLM context)
            current_store_lines.append(raw_line)
            continue

        # Regular content line – belongs to current store if one is open
        if current_store_lines:
            current_store_lines.append(raw_line)

    # Flush any trailing store at end of file
    flush_current_store()

    return docs


def build_retriever_from_markdown(file_path: str) -> Any:
    """
    Build a retriever from a mall markdown file with semantic, store-level chunks.

    Steps:
    1. Load raw markdown
    2. Parse into one Document per store with rich metadata
    3. Optionally sub-chunk long store descriptions
    4. Create embeddings with OpenAIEmbeddings (text-embedding-ada-002)
    5. Persist to Chroma DB in "chromadb" directory
    6. Return an MMR retriever (k=8, fetch_k=20)

    Args:
        file_path: Path to the markdown file

    Returns:
        A retriever object (vectorstore.as_retriever)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )

    # 1. Load raw markdown
    markdown_text = path.read_text(encoding="utf-8")
    if not markdown_text.strip():
        raise ValueError(f"No content found in markdown file: {file_path}")

    # 2. Parse into per-store Documents with metadata
    store_docs = _parse_store_documents(markdown_text)
    if not store_docs:
        raise ValueError(
            f"No store entries parsed from markdown file: {file_path}. "
            "Check the document format (expected 'Store name:' lines)."
        )

    # 3. Semantic chunking per store (only for long descriptions)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    chunks = text_splitter.split_documents(store_docs)

    if not chunks:
        raise ValueError(f"No chunks created from store documents: {file_path}")

    # Include category, sub_category, and tags in embedding text for better semantic matching
    for doc in chunks:
        category = doc.metadata.get("category", "")
        sub_category = doc.metadata.get("sub_category", "")
        tags = doc.metadata.get("tags", "")  # tags is now a comma-separated string
        
        # Build additional context string
        context_parts = []
        if category:
            context_parts.append(f"Category: {category}")
        if sub_category:
            context_parts.append(f"Subcategory: {sub_category}")
        if tags:
            context_parts.append(f"Tags: {tags}")
        
        # Append context to page_content so it's included in embeddings
        if context_parts:
            doc.page_content = f"{doc.page_content}\n" + "\n".join(context_parts)

    # 4. Create embeddings using text-embedding-ada-002
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # 5. Ensure chromadb directory exists for persistent storage
    chroma_db_dir = Path("chromadb")
    chroma_db_dir.mkdir(exist_ok=True)

    # Use a fixed collection name so we can overwrite contents on each upload
    collection_name = "giga_mall_rag"

    # 5a. If collection already exists, clear its contents instead of deleting files
    try:
        existing_vs = Chroma(
            persist_directory=str(chroma_db_dir),
            embedding_function=embeddings,
            collection_name=collection_name,
        )
        # delete all existing docs in this collection
        existing_vs.delete(where={})
    except Exception:
        # If collection doesn't exist yet, ignore
        pass

    # 6. Vector store with persistent Chroma DB for the new document
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_db_dir),
        collection_name=collection_name,
    )

    # MMR retriever for diverse results with better recall
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20},
    )

    return retriever


def load_existing_retriever() -> Any:
    """
    Load an existing Chroma vector store from the chromadb directory.
    
    This function loads the persisted Chroma DB instead of creating a new one.
    Use this when querying (not when uploading/creating).
    
    Returns:
        A retriever object (vectorstore.as_retriever)
        
    Raises:
        ValueError: If OpenAI API key is not set or chromadb directory doesn't exist
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    
    # Check if chromadb directory exists
    chroma_db_dir = Path("chromadb")
    if not chroma_db_dir.exists():
        raise ValueError(
            "Chroma DB directory not found. "
            "Please upload a markdown file via /rag/upload first."
        )
    
    # Create embeddings (must match the model used during creation)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Load existing Chroma vector store for the same collection
    vectorstore = Chroma(
        persist_directory=str(chroma_db_dir),
        embedding_function=embeddings,
        collection_name="giga_mall_rag",
    )
    
    # MMR retriever for diverse results
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20},
    )
    
    return retriever