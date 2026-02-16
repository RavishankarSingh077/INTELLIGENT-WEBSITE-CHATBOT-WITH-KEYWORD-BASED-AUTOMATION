#!/usr/bin/env python3
"""
FASC.AI Chatbot with ChromaDB RAG System
Complete implementation with multi-URL crawling and vector database
"""

import os
import logging
import hashlib
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Core libraries
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import uvicorn
import requests
import httpx
from bs4 import BeautifulSoup
import re

# Task scheduling
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Rate limiting and security
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Vector database and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Intent classification
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available, skipping Hugging Face intent classification")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "fascai_content"
EMBEDDING_MODEL = "all-mpnet-base-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MAX_URLS = 20
RELEVANCE_THRESHOLD = 1.0  # Maximum distance for relevant results (lower is better, stricter matching)

# Priority queries that must be pre-ingested to avoid on-demand scraping latency
PRIORITY_PRELOAD_QUERIES = [
    # AI & Automation
    "fasc ai automation services",
    "fasc ai ai solutions",
    "ai company fasc ai",
    "ai implementation services fasc ai",
    "intelligent automation fasc ai",
    # ERP & CRM
    "fasc ai erp implementation",
    "fasc ai crm services",
    "enterprise resource planning fasc ai",
    "customer relationship management fasc ai",
    # Cloud & IoT
    "cloud transformation services fasc ai",
    "cloud hosting fasc ai",
    "iot solutions fasc ai",
    # Pricing & Contact
    "fasc ai pricing",
    "fasc ai cost estimate",
    "how to contact fasc ai",
    "fasc ai support email",
    # Portfolio & Clients
    "fasc ai projects portfolio",
    "fasc ai client success",
    "fasc ai case studies",
    # General service info
    "what services does fasc ai offer",
    "fasc ai digital transformation services"
]

AI_AUTOMATION_QUERY_KEYWORDS = {
    "ai company",
    "ai service",
    "ai services",
    "ai solution",
    "ai solutions",
    "ai implementation",
    "ai implementations",
    "artificial intelligence",
    "machine learning",
    "ml",
    "deep learning",
    "neural network",
    "neural networks",
    "computer vision",
    "machine vision",
    "image processing",
    "object detection",
    "pattern recognition",
    "natural language processing",
    "nlp",
    "conversational ai",
    "virtual agent",
    "virtual assistant",
    "chatbot",
    "intelligent automation",
    "ai automation",
    "automation",
    "hyperautomation",
    "smart automation",
    "cognitive automation",
    "generative ai",
    "foundation model",
    "foundation models",
    "large language model",
    "large language models",
    "llm",
    "llms",
    "robotic process automation",
    "rpa",
    "autonomous systems",
    "edge ai",
}

# API Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Secure API keys from environment variables (supports rotation)
_raw_groq_keys = os.getenv("GROQ_API_KEYS", "")
GROQ_API_KEYS = [key.strip() for key in _raw_groq_keys.split(",") if key.strip()]

if not GROQ_API_KEYS:
    single_key = os.getenv("GROQ_API_KEY")
    if single_key and single_key.strip():
        GROQ_API_KEYS = [single_key.strip()]

if not GROQ_API_KEYS:
    DEFAULT_GROQ_KEYS = []
    GROQ_API_KEYS = [key for key in DEFAULT_GROQ_KEYS if key]

if not GROQ_API_KEYS:
    raise ValueError("Groq API key(s) not configured. Set GROQ_API_KEYS or GROQ_API_KEY before running the application.")

GROQ_API_KEYS = list(dict.fromkeys(GROQ_API_KEYS))

# Backwards compatibility constant (first key)
GROQ_API_KEY = GROQ_API_KEYS[0]

_current_groq_key_index = 0
_groq_key_lock = asyncio.Lock()


def _mask_api_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"


async def _get_active_groq_key() -> str:
    async with _groq_key_lock:
        return GROQ_API_KEYS[_current_groq_key_index]


async def _advance_groq_key() -> str:
    global _current_groq_key_index
    async with _groq_key_lock:
        _current_groq_key_index = (_current_groq_key_index + 1) % len(GROQ_API_KEYS)
        return GROQ_API_KEYS[_current_groq_key_index]


async def _call_groq_with_messages(
    messages: List[Dict[str, str]],
    temperature: float = 0.5,
    max_tokens: int = 100
) -> Optional[str]:
    """Utility to call Groq with rotation support and return the cleaned content."""
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    attempts = 0
    response: Optional[httpx.Response] = None

    async with httpx.AsyncClient(timeout=10.0) as client:
        while attempts < len(GROQ_API_KEYS):
            active_key = await _get_active_groq_key()
            headers = {
                "Authorization": f"Bearer {active_key}",
                "Content-Type": "application/json",
            }

            try:
                response = await client.post(GROQ_API_URL, headers=headers, json=data)
            except httpx.RequestError as exc:
                logger.error(f"Groq API request error with key {_mask_api_key(active_key)}: {exc}")
                attempts += 1
                await _advance_groq_key()
                continue

            if response.status_code == 200:
                try:
                    result = response.json()
                    ai_reply = result["choices"][0]["message"]["content"]
                    return strip_markdown(ai_reply.strip())
                except (KeyError, IndexError, ValueError) as exc:
                    logger.error(f"Unexpected Groq response structure: {exc}")
                    return None

            if response.status_code in (401, 403, 429):
                logger.warning(
                    f"Groq API key {_mask_api_key(active_key)} returned status {response.status_code}. Rotating to next key."
                )
                attempts += 1
                await _advance_groq_key()
                continue

            logger.error(
                f"Groq API error {response.status_code} with key {_mask_api_key(active_key)}: {response.text}"
            )
            break

    return None

# Safe fallback response used when sanitizing replies
SAFE_FALLBACK_REPLY = "We provide cloud computing, ERP, CRM, AI solutions, and IoT services."
AI_AUTOMATION_FALLBACK = (
    "We deliver AI-driven automation—machine learning models, conversational AI, computer vision, and "
    "intelligent process automation—to strengthen your operations. Let me know the scenario you have in mind and I'll map the best approach."
)

SOFT_NEGATIVE_PHRASES = [
    "you are mad",
    "you're mad",
    "you are irritating",
    "you're irritating",
    "you are annoying",
    "you're annoying",
    "you are rude",
    "you're rude",
    "i am not your client",
    "i'm not your client",
    "i am not your customer",
    "i'm not your customer",
    "i don't need your help",
    "i dont need your help",
    "i do not need your help",
]

# --------------------------------------------------------------------------------------------------
# Response sanitization utilities
# --------------------------------------------------------------------------------------------------
def sanitize_response_text(reply: Optional[str]) -> str:
    """Keep friendly statements while stripping questions and forbidden phrases."""
    if not reply:
        return SAFE_FALLBACK_REPLY

    reply = reply.strip()
    if not reply:
        return SAFE_FALLBACK_REPLY

    original_reply = reply
    sentence_split_pattern = re.compile(r'(?<=[.!?])\s+')
    question_phrase_patterns = [
        r'would you like to',
        r'do you want to',
        r'can i help you',
        r'what would you like',
        r'how can i assist',
        r'is there anything',
        r'are you exploring',
        r'what challenges',
        r"what's on your mind",
        r'what would you like to explore',
        r'how can we support',
        r'want to hear about',
        r'let me ask',
        r"since you're here",
        r'is there anything about',
        r'would you like to know',
        r'would you like to learn',
        r'would you like to discuss',
        r'would you like to explore',
        r'would you like more',
        r'would you like additional',
        r'would you like to hear',
        r'would you like to find out',
        r'can we support',
        r'how can we',
        r'what do you need',
        r'what do you want',
    ]

    sentences = sentence_split_pattern.split(reply)
    kept_sentences: List[str] = []

    for sentence in sentences:
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue

        sentence_lower = sentence_clean.lower()
        if '?' in sentence_lower:
            continue
        if any(re.search(pattern, sentence_lower) for pattern in question_phrase_patterns):
            continue

        kept_sentences.append(sentence_clean.rstrip('.!?'))
        if len(kept_sentences) >= 2:
            break

    if not kept_sentences:
        logger.warning(f"Reply removed during sanitization. Original: {original_reply[:120]}")
        sanitized = SAFE_FALLBACK_REPLY
    else:
        sanitized = '. '.join(kept_sentences).strip()
        if not sanitized.endswith('.'):
            sanitized = f"{sanitized}."

    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    correct_number = "+91-9958755444"
    wrong_number_patterns = [
        r'\+91\s*11\s*4567\s*8900',
        r'\+91\s*11-4567-8900',
        r'\+91\s*11\s*4567-8900',
        r'\+91\s*11-4567\s*8900',
        r'11\s*4567\s*8900',
        r'11-4567-8900'
    ]
    for pattern in wrong_number_patterns:
        sanitized = re.sub(pattern, correct_number, sanitized, flags=re.IGNORECASE)

    phone_pattern = r'\+91[\s-]?\d{2}[\s-]?\d{4}[\s-]?\d{4}'
    found_numbers = re.findall(phone_pattern, sanitized)
    for found_num in found_numbers:
        normalized_found = re.sub(r'[\s-]', '', found_num)
        normalized_correct = re.sub(r'[\s-]', '', correct_number)
        if normalized_found != normalized_correct:
            sanitized = sanitized.replace(found_num, correct_number)

    if '?' in sanitized:
        sanitized = sanitized.replace('?', '.').strip()

    if not sanitized:
        sanitized = SAFE_FALLBACK_REPLY

    return sanitized


def extract_context_snippet(search_results: Optional[List[Dict[str, Any]]], max_words: int = 25) -> Optional[str]:
    """Extract a short informative snippet from search results to enrich terse replies."""
    if not search_results:
        return None
    for result in search_results:
        if not isinstance(result, dict):
            continue
        text = result.get('content', '')
        if not text:
            continue
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            cleaned = re.sub(r'\s+', ' ', sentence).strip()
            if not cleaned:
                continue
            if len(cleaned.split()) < 6:
                continue
            if len(cleaned.split()) > max_words:
                cleaned = ' '.join(cleaned.split()[:max_words]) + '...'
            cleaned = cleaned.replace('?', '.')
            return cleaned
    return None

# FastAPI app
app = FastAPI(title="FASC.AI ChromaDB RAG Chatbot", version="2.0.0")

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration - Secure setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://fascai.com",
        "https://www.fascai.com", 
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "file://",
        "null"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Global variables for caching
chroma_client = None
embedding_model = None
intent_classifier = None
# Embedding cache to avoid regenerating same query embeddings
embedding_cache = {}

# Global scheduler for periodic tasks
scheduler = None

# In-memory conversation storage
# Structure: { 
#   "session_id": {
#     "conversations": [{"role": "user", "content": "message"}, {"role": "assistant", "content": "reply"}],
#     "language": "english" or "hindi"
#   }
# }
conversation_sessions = {}

# Language detection function
def detect_language(message: str) -> str:
    """Detect if message is in Hindi or English based on comprehensive word list"""
    if not message or not message.strip():
        return 'english'  # Default for empty messages
    
    message = message.strip()
    message_lower = message.lower()
    
    # Devanagari script range (Hindi characters)
    devanagari_pattern = re.compile(r'[\u0900-\u097F]')
    
    # Check if message contains Devanagari script - if yes, it's Hindi
    if bool(devanagari_pattern.search(message)):
        return 'hindi'
    
    # Check for common English greetings first (before Hindi word detection)
    english_greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'hiya', 'howdy', 'greetings'
    ]
    
    # Check for exact matches or short greeting phrases
    if message_lower in english_greetings:
        return 'english'
    
    # Check if message starts with an English greeting and has 3 or fewer words
    for greeting in english_greetings:
        if message_lower.startswith(greeting) and len(message_lower.split()) <= 3:
            return 'english'
    # Comprehensive list of common Hindi words in Roman script
    hindi_words = {
        # Common verbs and conjugations
        'ho', 'hai', 'hain', 'hun', 'hoon', 'hoa', 'hoga', 'hogi', 'honge',
        'karta', 'karti', 'karte', 'kiya', 'kiye', 'kiyi', 'karega', 'karegi', 'karenge',
        'kar', 'karo', 'kariye', 'karein', 'kare',
        'bolo', 'boliye', 'bolte', 'bola', 'bol', 'kaho',
        'dekh', 'dekho', 'dekhna', 'dekhte', 'dekha', 'dekhega',
        'banaya', 'banai', 'banate', 'banega', 'banegi',
        'liya', 'li', 'le', 'lete', 'leta', 'leti', 'letiye', 'lena', 'lene',
        'gaya', 'gayi', 'gaye', 'jao', 'ja', 'jaate', 'jaata', 'jaati', 'jana', 'jane',
        'aayega', 'aayegi', 'aayenge', 'aao', 'aa', 'aate', 'aata', 'aati', 'aana', 'aane',
        'diya', 'diye', 'dete', 'deta', 'deti', 'dene', 'de', 'do', 'dena', 'dene',
        
        # Pronouns
        'tum', 'tumhare', 'tumhari', 'tumhara', 'tumhe', 'tumko', 'tumse',
        'aap', 'aapka', 'aapki', 'aapke', 'aapko', 'aapse',
        'main', 'mera', 'meri', 'mere', 'mujhe', 'mujhse', 'mujhko',
        'woh', 'uska', 'uski', 'uske', 'use', 'usse',
        'yeh', 'ye', 'iske', 'iski', 'iska', 'ise', 'isse', 'isne',
        'hum', 'hamara', 'hamari', 'hamare', 'humhe', 'humko', 'hamse',
        'kaun', 'kaunse', 'kaunsi', 'kaunka', 'kiska', 'kiski', 'kiske',
        
        # Question words
        'kya', 'kyun', 'kahan', 'kab', 'kaise', 'kitna', 'kitni', 'kitne', 'kisko', 'kisne',
        'kyon', 'kaun', 'kaise', 'kabhi', 'kab',
        
        # Common prepositions/particles
        'ko', 'se', 'par', 'mein', 'ne', 'toh', 'bhi', 'na', 'nahi', 'nahin',
        'tak', 'ke', 'ki', 'ka', 'kar', 'ke', 'hie',
        
        # Common adjectives/adverbs
        'acha', 'achcha', 'accha', 'badhiya', 'theek', 'sahi', 'galat', 'bura', 'bura',
        'bahut', 'bohot', 'bhot', 'zyada', 'kam', 'khub', 'kaafi',
        'abhi', 'ab', 'phir', 'fir', 'tab', 'toh',
        
        # Common nouns
        'ghar', 'ghar', 'kaam', 'kaam', 'log', 'insaan', 'admi', 'aurat',
        'dost', 'dosti', 'pyar', 'mohabbat', 'khushi', 'dukh', 'gham',
        
        # Time words
        'aaj', 'kal', 'parso', 'roz', 'hamesha', 'kabhi', 'abhi', 'phir', 'fir',
        'subah', 'shaam', 'raat', 'din', 'mahina', 'saal', 'samay', 'waqt',
        
        # Greetings and common phrases
        'namaste', 'namaskar', 'dhanyawad', 'shukriya', 'kripya', 'maaf',
        'hain', 'sab', 'hota', 'hoti', 'hote',
        
        # Common words
        'bilkul', 'zaroor', 'pakka', 'sach', 'sacchi', 'sach',
        'acha', 'theek', 'sahi', 'galat', 'bura', 'sahi',
        'bhi', 'sirf', 'bas', 'abhi', 'phir',
        'mai', 'tumhari', 'mera', 'tera', 'hamara',
        'nahi', 'han', 'haan', 'na', 'ji', 'ji',
        
        # Action words
        'karo', 'karein', 'karna', 'karne', 'kar', 'karne',
        'bolo', 'boliye', 'bolna', 'bolne', 'bol',
        'dekh', 'dekho', 'dekhna', 'dekhne',
        
        # More common words
        'usne', 'unhone', 'unke', 'unka', 'unki',
        'raha', 'rahi', 'rahe', 'rahon', 'rahun',
        'chahiye', 'chahiye', 'chahta', 'chahti', 'chahte',
        'milna', 'milne', 'repository', 'project', 'repository', 'budget',
    }
    
    # Extract words from message
    words = re.findall(r'\b\w+\b', message_lower)
    
    if not words:
        return 'english'  # No words found
    
    # Count Hindi words
    hindi_count = sum(1 for word in words if word in hindi_words)
    total_words = len(words)
    
    # If 30% or more words are Hindi, it's Hindi
    if hindi_count > 0 and (hindi_count / total_words) >= 0.3:
        return 'hindi'
    elif hindi_count > total_words - hindi_count:
        return 'hindi'
    else:
        return 'english'

def is_ai_automation_query(message: str) -> bool:
    """Quick heuristic to detect AI / automation themed queries."""
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in AI_AUTOMATION_QUERY_KEYWORDS)


def is_soft_negative_message(message: str) -> bool:
    """Identify negative/frustrated phrasing that should still be answered by the LLM."""
    message_lower = message.lower()
    return any(phrase in message_lower for phrase in SOFT_NEGATIVE_PHRASES)

# Hugging Face Intent Classifier Functions
def get_intent_classifier():
    """Load and return the Hugging Face intent classifier model"""
    global intent_classifier
    
    if intent_classifier is not None:
        return intent_classifier
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers library not available, skipping intent classifier")
        return None
    
    try:
        logger.info("Loading Hugging Face intent classifier model (DistilBERT)...")
        # Using DistilBERT for faster zero-shot classification (3-4x faster, 6x smaller)
        intent_classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",  # Fast and lightweight intent classification
            device=-1  # Use CPU
        )
        logger.info("Intent classifier model loaded successfully")
        return intent_classifier
    except Exception as e:
        logger.error(f"Error loading intent classifier: {e}")
        intent_classifier = None
        return None
def detect_intent_with_hf(message: str) -> dict:
    """
    Detect intent using Hugging Face model with zero-shot classification
    
    Returns:
        dict with 'intent' and 'confidence' keys, or None if model unavailable
    """
    try:
        classifier = get_intent_classifier()
        if not classifier:
            return None
        
        # Define intent candidates with descriptive phrases based on pattern checks
        # These descriptions include keywords from pattern-based functions for better accuracy
        intent_candidates = [
            # Greeting - from is_greeting() patterns
            "user is greeting or saying hello with words like hi, hello, hey, good morning, good afternoon, good evening, namaste, namaskar, salam, hiya, howdy, greetings, kaise ho, kaise hain",
            
            # Goodbye - from is_goodbye() patterns
            "user is saying goodbye or ending conversation with words like bye, goodbye, see you, later, end chat, exit, quit, alvida, phir milte hain",
            
            # Project inquiry - from is_project_intent() patterns
            "user wants to start a new project, work with company, need project help, want to work with you, planning a project, need help with project",
            
            # ERP-specific inquiry
            "user is asking about ERP systems, what is ERP, how can ERP help, ERP implementation, ERP services, ERP solutions, enterprise resource planning",
            
            # CRM-specific inquiry
            "user is asking about CRM systems, what is CRM, how can CRM help, CRM implementation, CRM services, CRM solutions, customer relationship management, set up CRM, help with CRM",
            
            # Cloud/Hosting inquiry
            "user is asking about cloud hosting, cloud computing, cloud services, cloud solutions, hosting services, AWS, Azure, Google Cloud, cloud migration, cloud setup",
            
            # IoT inquiry
            "user is asking about IoT solutions, Internet of Things, IoT implementation, IoT services, IoT devices, IoT platforms",
            
            # AI solutions inquiry
            "user is asking about AI solutions, artificial intelligence, machine learning, AI implementation, AI services, chatbot development, AI platforms",
            
            # General service inquiry - from is_service_inquiry() and is_service_info_query() patterns
            "user is asking about IT services offered like cloud computing, ERP systems, CRM platforms, AI solutions, IoT services, what services do you offer, tell me about your services",
            
            # Business info - from is_business_info_query() patterns
            "user is asking about business information like success rate, customer satisfaction, client satisfaction, team size, how many employees, experience, years of experience, portfolio, case studies, clients, customer base",
            
            # Existing customer - from is_existing_customer_query() patterns
            "user is an existing customer asking about their project, I am a client, I have a project with you, my project name is",
            
            # Personal introduction - from is_personal_introduction() patterns
            "user is introducing themselves by sharing their name, my name is, I am, call me",
            
            # Bot identity - from is_bot_identity_question() patterns
            "user is asking about bot's identity or name like who are you, what is your name, tell me your name, apka naam kya hai",
            
            # Contact inquiry - from is_contact_query() patterns
            "user is asking for contact information like location, address, where are you, office, email, phone, how to contact, get in touch, talk to someone",
            
            # Capability question - from is_capability_question() patterns (GENERIC ONLY, not service-specific)
            "user is asking GENERICALLY about chatbot capabilities like what can you do, what are your capabilities, what can you assist with, BUT NOT asking about specific services like ERP or CRM",
            
            # Complaint - from is_complaint() patterns
            "user has a complaint or is dissatisfied like not happy, unhappy, disappointed, poor service, bad service, complaint, problem with service",
            
            # Help request - GENERIC help only (not service-specific)
            "user needs GENERIC help or has questions like help, need help, can you help, I need assistance, how can you help me, BUT NOT asking about specific services like ERP, CRM, Cloud, or IoT",
            
            # Pricing inquiry - from is_pricing_query() patterns (with negative exclusions)
            "user is asking about pricing information like price, pricing, cost, how much, expensive, cheap, fees, charge, payment, quote, estimate BUT NOT asking about success rate, conversion rate, performance metrics, or business statistics",
            
            # Off topic - from is_off_topic() patterns
            "user is asking something completely unrelated to company IT services like movies, weather, recipes, other companies like Google or Flipkart, unrelated topics"
        ]
        
        # Create mapping from descriptive labels to simple intent names
        intent_mapping = {
            "user is greeting or saying hello": "greeting",
            "user is saying goodbye or ending": "goodbye",
            "user wants to start a new project": "project_inquiry",
            "user is asking about ERP": "erp_inquiry",
            "user is asking about CRM": "crm_inquiry",
            "user is asking about cloud": "cloud_inquiry",
            "user is asking about IoT": "iot_inquiry",
            "user is asking about AI": "ai_inquiry",
            "user is asking about IT services": "service_inquiry",
            "user is asking about business information": "business_info",
            "user is an existing customer": "existing_customer",
            "user is introducing themselves": "personal_introduction",
            "user is asking about bot's identity": "bot_identity",
            "user is asking for contact information": "contact_inquiry",
            "user is asking GENERICALLY about chatbot capabilities": "capability_question",
            "user has a complaint": "complaint",
            "user needs GENERIC help": "help_request",
            "user is asking about pricing information": "pricing_inquiry",
            "user is asking something completely unrelated": "off_topic"
        }
        
        # Dynamic confidence thresholds based on intent type (LOWERED for better coverage)
        intent_thresholds = {
            "greeting": 0.55,
            "goodbye": 0.55,
            "help_request": 0.55,
            "contact_inquiry": 0.55,
            "erp_inquiry": 0.50,
            "crm_inquiry": 0.50,
            "cloud_inquiry": 0.50,
            "iot_inquiry": 0.50,
            "ai_inquiry": 0.50,
            "service_inquiry": 0.50,
            "pricing_inquiry": 0.55,
            "business_info": 0.55,
            "project_inquiry": 0.55,
            "bot_identity": 0.55,
            "capability_question": 0.55,
            "existing_customer": 0.55,
            "personal_introduction": 0.55,
            "complaint": 0.60,
            "off_topic": 0.65
        }
        
        # Classify the message
        result = classifier(message, intent_candidates)
        
        if result and len(result['labels']) > 0:
            top_intent_label = result['labels'][0]
            confidence = result['scores'][0]
            
            # Map descriptive label back to simple intent name
            detected_intent = None
            # Check technical service intents first (more specific)
            if "erp" in top_intent_label.lower() and "asking about" in top_intent_label.lower():
                detected_intent = "erp_inquiry"
            elif "crm" in top_intent_label.lower() and "asking about" in top_intent_label.lower():
                detected_intent = "crm_inquiry"
            elif ("cloud" in top_intent_label.lower() or "hosting" in top_intent_label.lower()) and "asking about" in top_intent_label.lower():
                detected_intent = "cloud_inquiry"
            elif "iot" in top_intent_label.lower() and "asking about" in top_intent_label.lower():
                detected_intent = "iot_inquiry"
            elif ("ai" in top_intent_label.lower() or "artificial intelligence" in top_intent_label.lower()) and "asking about" in top_intent_label.lower():
                detected_intent = "ai_inquiry"
            else:
                # Fallback to general mapping
                for label_key, intent_name in intent_mapping.items():
                    if label_key in top_intent_label.lower():
                        detected_intent = intent_name
                        break
            
            # If no mapping found, try to extract from label
            if not detected_intent:
                # Extract first few words as fallback
                if "greeting" in top_intent_label.lower():
                    detected_intent = "greeting"
                elif "goodbye" in top_intent_label.lower():
                    detected_intent = "goodbye"
                elif "erp" in top_intent_label.lower():
                    detected_intent = "erp_inquiry"
                elif "crm" in top_intent_label.lower():
                    detected_intent = "crm_inquiry"
                elif "cloud" in top_intent_label.lower() or "hosting" in top_intent_label.lower():
                    detected_intent = "cloud_inquiry"
                elif "iot" in top_intent_label.lower():
                    detected_intent = "iot_inquiry"
                elif "ai" in top_intent_label.lower() or "artificial intelligence" in top_intent_label.lower():
                    detected_intent = "ai_inquiry"
                elif "business information" in top_intent_label.lower():
                    detected_intent = "business_info"
                elif "pricing" in top_intent_label.lower():
                    detected_intent = "pricing_inquiry"
                elif "service" in top_intent_label.lower():
                    detected_intent = "service_inquiry"
                elif "project" in top_intent_label.lower():
                    detected_intent = "project_inquiry"
                else:
                    detected_intent = "off_topic"  # Default fallback
            
            # Get threshold for this intent
            threshold = intent_thresholds.get(detected_intent, 0.75)
            
            logger.info(f"HF Intent detected: {detected_intent} with confidence: {confidence:.2f} (threshold: {threshold:.2f})")
            
            return {
                'intent': detected_intent,
                'confidence': confidence,
                'threshold': threshold,
                'all_intents': list(zip(result['labels'], result['scores']))
            }
        
        return None
    
    except Exception as e:
        logger.error(f"Error in Hugging Face intent detection: {e}")
        return None
# Query classification functions
def get_off_topic_category(message: str) -> str:
    """Categorize off-topic queries for appropriate responses"""
    message_lower = message.lower().strip()
    
    if is_soft_negative_message(message):
        return None
    
    # Check if it's a Hindi service query (should be allowed)
    hindi_service_keywords = [
        'services', 'solutions', 'cloud', 'erp', 'crm', 'ai', 'iot', 'fasc ai',
        'aapke', 'ke bare', 'batao', 'kya hain', 'provide', 'karte hain'
    ]
    
    # If Hindi message contains service-related keywords, don't mark as off-topic
    if any(keyword in message_lower for keyword in hindi_service_keywords):
        return None
    
    # Abusive or inappropriate language patterns
    abusive_patterns = [
        'pagal', 'idiot', 'stupid', 'harami', 'fool', 'nonsense', 'shut up',
        'fuck', 'shit', 'damn', 'hate', 'useless', 'garbage', 'trash',
        'chup', 'chup sale', 'sale', 'kutta', 'kutte', 'bevakoof', 'ullu'
    ]
    
    # Other company keywords
    other_company_keywords = [
        'google', 'microsoft', 'amazon', 'facebook', 'meta', 'apple', 
        'netflix', 'tesla', 'ibm', 'oracle', 'salesforce'
    ]
    
    # Job-related keywords for other companies
    job_other_company = [
        'google job', 'job in google', 'job at google', 'microsoft job', 'amazon job',
        'facebook job', 'apple job', 'work at google', 'work at microsoft'
    ]
    
    # Unrelated topics
    unrelated_keywords = [
        'bhojpuri', 'recipe', 'cooking', 'movie', 'song', 'game', 'sport',
        'weather', 'news', 'politics', 'celebrity', 'fashion', 'shopping',
        'love', 'dating', 'relationship', 'marriage'
    ]
    
    # Generic unrelated
    generic_unrelated = ['close it', 'closeit', 'stop', 'don\'t show', 'hide']
    
    # Check for abusive language
    for pattern in abusive_patterns:
        if pattern in message_lower:
            return 'abusive'
    
    # Check for job queries at other companies
    for keyword in job_other_company:
        if keyword in message_lower:
            return 'job_other_company'
    
    # Check for other company mentions
    for keyword in other_company_keywords:
        if keyword in message_lower and 'job' not in message_lower:
            return 'other_company'
    
    # Check for unrelated topics
    for keyword in unrelated_keywords:
        if keyword in message_lower:
            return 'unrelated'
    
    # Check for generic unrelated
    if message_lower in generic_unrelated:
        return 'unrelated'
    
    return None

def is_how_are_you_question(message: str) -> bool:
    """Check if message is asking 'how are you' type questions"""
    message_lower = message.lower().strip()
    how_are_you_patterns = [
        'how are you', 'how are you doing', 'how do you do', 'how\'s it going',
        'how\'s your day', 'how\'s everything', 'how\'s life', 'what\'s up',
        'how\'s work', 'how\'s things', 'how are things', 'how\'s your day going',
        'kaise ho', 'kaise hain', 'aap kaise hain', 'tum kaise ho',
        'aap kaise hain', 'kaise chal raha hai', 'sab theek hai'
    ]
    
    for pattern in how_are_you_patterns:
        if pattern in message_lower:
            return True
    
    return False

def is_emotional_expression(message: str) -> bool:
    """Check if user is expressing emotions or feelings"""
    message_lower = message.lower().strip()
    
    emotional_patterns = [
        'i am happy', 'i am sad', 'i am frustrated', 'i am angry', 'i am upset',
        'i am excited', 'i am worried', 'i am disappointed', 'i am pleased',
        'i am annoyed', 'i am good', 'i am bad', 'i am fine', 'i am alright',
        'i am perfect', 'i am excellent', 'i am wonderful', 'i am great',
        'i am terrible', 'i am awful', 'i am amazing', 'i am fantastic',
        'i am perform well', 'i am facing difficulties', 'i am upset with',
        'i am frustrated with', 'i am happy with', 'i am sad about',
        'i am excited about', 'i am worried about', 'i am disappointed with',
        'i am pleased with', 'i am angry about', 'i am annoyed with',
        'upset with', 'frustrated with', 'happy with', 'sad about',
        'excited about', 'worried about', 'disappointed with', 'pleased with',
        'angry about', 'annoyed with', 'excited for', 'worried for',
        'disappointed in', 'pleased about'
    ]
    
    return any(pattern in message_lower for pattern in emotional_patterns)

def is_user_doubt(message: str) -> bool:
    """Check if user is expressing doubt about chatbot's ability to help"""
    message_lower = message.lower().strip()
    
    doubt_patterns = [
        'i don\'t think you could help me', 'i don\'t think you can help',
        'you probably can\'t help', 'i doubt you can help', 'not sure you can help',
        'i don\'t think you could', 'i don\'t think you can', 'you can\'t help',
        'you probably can\'t', 'i doubt you', 'not sure you', 'you won\'t be able',
        'you might not be able', 'i\'m not sure you', 'i don\'t believe you',
        'you probably won\'t', 'you likely can\'t', 'you may not be able',
        'i don\'t think so', 'probably not', 'doubt it', 'not confident',
        'you\'re not helpful', 'you\'re not useful', 'you can\'t do it',
        'you won\'t understand', 'you don\'t know', 'you\'re not smart enough'
    ]
    
    return any(pattern in message_lower for pattern in doubt_patterns)

def is_help_request(message: str) -> bool:
    """Check if user is asking for help or facing difficulties - simplified to core keywords only"""
    message_lower = message.lower().strip()
    
    # FIRST: Check for minimal negative patterns - if found, return False (let RAG handle)
    negative_patterns = [
        "don't need", "dont need", "do not need",
        "don't want", "dont want", "do not want",
        "i don't need", "i dont need", "i do not need",
        "i am not your client", "i am not your customer", "i'm not your client"
    ]
    
    if any(pattern in message_lower for pattern in negative_patterns):
        return False  # Negative case - let RAG handle with human-like responses
    
    # SECOND: Check for core help keywords only (5-10 keywords)
    # Service-specific queries will be handled by project_manager or RAG flow
    help_keywords = [
        'help', 'assistance', 'support', 'guidance', 'trouble',
        'stuck', 'confused', 'problems', 'issues', 'difficulties'
    ]
    
    return any(keyword in message_lower for keyword in help_keywords)

def is_new_user_indication(message: str) -> bool:
    """Check if user is indicating they are new/first time"""
    message_lower = message.lower().strip()
    
    new_user_patterns = [
        'first time', 'new here', 'just started', 'new user', 'new customer',
        'i am coming', 'coming on', 'first visit', 'never been', 'never used',
        'don\'t know', 'don\'t have', 'don\'t think', 'could help',
        'how you found', 'found my project', 'i am coming on fasc ai first time',
        'coming on fasc ai first time', 'first time here', 'never used this',
        'never been here', 'just started using', 'new to this', 'new to fasc',
        'first visit to fasc', 'never used fasc', 'never been on fasc'
    ]
    
    return any(pattern in message_lower for pattern in new_user_patterns)
def clear_project_context(session_id: str):
    """Clear project context for a session"""
    if session_id in conversation_sessions:
        if 'project_context' in conversation_sessions[session_id]:
            conversation_sessions[session_id]['project_context'] = {}
            logger.info(f"Cleared project context for session: {session_id}")

def is_personal_introduction(message: str) -> bool:
    """Check if message contains personal introduction/name sharing"""
    message_lower = message.lower().strip()
    
    # First check if it's dissatisfaction - if so, don't treat as name introduction
    if is_dissatisfaction(message):
        return False
    
    # COMPREHENSIVE CONTEXT ANALYSIS - Check this FIRST before any pattern matching
    def has_context_words(text):
        """Check if text contains context words that indicate it's not a name introduction"""
        context_indicators = [
            # Action words and verbs
            'looking', 'searching', 'finding', 'seeking', 'asking', 'telling',
            'wanting', 'needing', 'trying', 'doing', 'going', 'coming',
            'working', 'playing', 'running', 'walking', 'sitting', 'standing',
            'eating', 'drinking', 'sleeping', 'waking', 'buying', 'selling',
            'helping', 'using', 'opening', 'closing', 'starting', 'stopping',
            'beginning', 'ending', 'finishing', 'getting', 'making', 'taking',
            'giving', 'seeing', 'knowing', 'thinking', 'having', 'being',
            # Purpose and intention words
            'for', 'about', 'regarding', 'concerning', 'to', 'with', 'by',
            'want', 'need', 'require', 'interested', 'curious', 'wondering',
            # Technology and service words
            'website', 'chatbot', 'bot', 'system', 'service', 'company', 'business',
            'project', 'work', 'job', 'task', 'problem', 'issue', 'question', 'answer',
            'solution', 'help', 'assistance', 'support', 'information', 'data',
            'crm', 'erp', 'ai', 'cloud', 'computing', 'development', 'software',
            # Comparison words
            'just like', 'similar to', 'for my', 'for your', 'for the', 'like you',
            'same as', 'like this', 'like that', 'as you', 'as me',
            # Common nouns that are not names
            'time', 'day', 'night', 'morning', 'evening', 'year', 'month', 'week',
            'place', 'location', 'area', 'city', 'country', 'world', 'earth',
            'thing', 'stuff', 'item', 'object', 'product', 'tool',
            'book', 'movie', 'music', 'food', 'water', 'money', 'price', 'cost'
        ]
        words = text.lower().split()
        return any(indicator in words for indicator in context_indicators)
    
    # If context words are detected, reject immediately
    if has_context_words(message_lower):
        return False
    
    # Check for emotional words that should not be treated as names
    # Include common misspellings
    emotional_words = ['frustrated', 'frustated', 'frustate', 'angry', 'sad', 'happy', 'excited', 'worried', 'disappointed', 'pleased', 'upset', 'annoyed', 'irritated']
    if any(word in message_lower for word in emotional_words):
        return False
    
    # Check for phone number patterns - these are not introductions
    import re
    phone_patterns = [
        r'\b\d{10}\b',  # 10 digits
        r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
        r'\b\+?\d{1,3}\s?\d{10}\b',  # +91 1234567890
        r'\b\d{5}\s?\d{5}\b'  # 12345 67890
    ]
    for pattern in phone_patterns:
        if re.search(pattern, message):
            return False
    
    # More specific name introduction patterns - require explicit introduction phrases
    intro_patterns = [
        'my name is', 'i\'m called', 'people call me', 'you can call me',
        'main hun', 'mera naam', 'mujhe kehte hain'
    ]
    
    # Check for name patterns - but be very strict
    for pattern in intro_patterns:
        if pattern in message_lower:
            # Additional validation: make sure it's not a complaint or question
            words = message_lower.split()
            # Exclude if contains negative words, questions, or action verbs
            excluded_words = ['not', 'bad', 'wrong', 'satisfied', 'happy', 'good', 'helpful', 
                            'asking', 'wondering', 'curious', 'want', 'need', 'looking', 
                            'interested', 'about', 'regarding', 'concerning', 'frustrated', 'frustated',
                            'angry', 'sad', 'excited', 'worried', 'disappointed', 'pleased', 'upset',
                            'on', 'at', 'in', 'to', 'for', 'with', 'by', 'from']
            if not any(word in excluded_words for word in words):
                return True
    
    # ENHANCED "i am" pattern handling - much stricter but supports full names
    if ('i am' in message_lower or 'i\'m' in message_lower):
        parts = message_lower.split()
        for i, part in enumerate(parts):
            if part in ['am', 'i\'m'] and i + 1 < len(parts):
                # Check for full names (up to 3 words after "i am")
                potential_name_parts = []
                for j in range(i + 1, min(i + 4, len(parts))):  # Check up to 3 words
                    potential_name_parts.append(parts[j])
                
                # Check if any of the potential name parts are context words
                potential_name_str = ' '.join(potential_name_parts)
                if has_context_words(potential_name_str):
                    continue
                
                # Check if all parts look like valid name components
                valid_name_parts = []
                for part in potential_name_parts:
                    # Skip common words that shouldn't be in names
                    if part in ['frustrated', 'frustated', 'frustate', 'angry', 'sad', 'happy', 'excited', 'worried', 
                               'disappointed', 'pleased', 'good', 'bad', 'right', 'wrong', 'upset', 'annoyed',
                               'looking', 'searching', 'finding', 'seeking', 'asking', 'telling', 'wanting', 'needing',
                               'website', 'chatbot', 'bot', 'system', 'service', 'company', 'business', 'project',
                               'work', 'job', 'task', 'problem', 'issue', 'question', 'answer', 'solution',
                               'help', 'assistance', 'support', 'information', 'for', 'about', 'regarding',
                               'just', 'like', 'similar', 'to', 'with', 'by', 'from', 'on', 'at', 'in',
                               'the', 'a', 'an', 'and', 'or', 'but', 'so', 'yet', 'nor', 'for', 'of',
                               'difficulties', 'perform', 'well', 'system', 'aren\'t', 'working', 'can\'t', 'find',
                               'facing', 'trying', 'going', 'wanting', 'needing', 'looking', 'searching',
                               'first', 'time', 'new', 'here', 'just', 'started', 'coming', 'found']:
                        break
                    
                    # Check if it looks like a valid name part (alphabetic, reasonable length)
                    if part.isalpha() and 2 <= len(part) <= 20:
                        valid_name_parts.append(part)
                    else:
                        break
                
                # If we have valid name parts, check if there are more words after
                if valid_name_parts:
                    remaining_words = parts[i + 1 + len(valid_name_parts):]
                    # If there are more words after the name, they should not be context words
                    if remaining_words and has_context_words(' '.join(remaining_words)):
                        continue
                    
                    # Final validation: ensure no context words in the entire sentence
                    if not has_context_words(message_lower):
                        return True
    
    return False

def is_name_recall_question(message: str) -> bool:
    """Check if user is asking about their own name"""
    message_lower = message.lower().strip()
    
    # More specific name recall patterns - avoid false positives
    name_recall_patterns = [
        'what is my name', 'what\'s my name', 'do you know my name', 
        'do you remember my name', 'what did i tell you my name', 
        'can you tell me my name', 'tell me my name',
        'mera naam kya hai', 'naam yaad hai', 'kya naam hai'
    ]
    
    for pattern in name_recall_patterns:
        if pattern in message_lower:
            return True
    
    # Special case for "my name" - only if it's a question or request
    if 'my name' in message_lower:
        # Check if it's part of a question or request
        if any(word in message_lower for word in ['what', 'tell', 'remember', 'know', 'recall']):
            return True
    
    return False

def is_personality_question(message: str) -> bool:
    """Check if message is asking about personality/friendliness"""
    message_lower = message.lower().strip()
    personality_patterns = [
        'are you friendly', 'are you nice', 'are you helpful', 'are you good',
        'would you like to be my friend', 'can we be friends', 'be my friend',
        'are you a friend', 'do you like me', 'do you care', 'are you kind',
        'are you warm', 'are you personable', 'are you approachable',
        'are you welcoming', 'are you supportive', 'are you understanding',
        'are you patient', 'are you gentle', 'are you caring',
        'tum dost ban sakte ho', 'dost bano', 'aap dost hain', 'dost banoge',
        'aap achhe hain', 'aap friendly hain', 'aap helpful hain'
    ]
    
    for pattern in personality_patterns:
        if pattern in message_lower:
            return True
    
    return False
def get_how_are_you_response(user_language: str = 'english') -> str:
    """Generate natural response for 'how are you' questions"""
    import random
    
    if user_language == 'hindi':
        responses = [
            "Main bilkul theek hun! Dhanyawad puchhne ke liye. Main yahan hun aapki madad ke liye Fasc Ai ke IT solutions ke bare mein. Aap kya janna chahte hain?",
            "Main achha hun! Aapka dhanyawad. Main aapka AI assistant hun aur aapki IT services ke bare mein jaankari dene ke liye ready hun. Kya puchhna chahte hain?",
            "Main theek hun! Main yahan hun aapki help ke liye. Fasc Ai ke cloud computing, ERP, CRM, AI solutions ke bare mein koi sawal hai?"
        ]
    else:
        responses = [
            "I'm doing great, thank you for asking! I'm here to help you with Fasc Ai's IT solutions.",
            "I'm excellent! Thanks for asking. I'm ready to assist you with information about our cloud computing, ERP, CRM, and AI solutions.",
            "I'm wonderful! I'm here to help you learn about Fasc Ai's services."
        ]
    
    return random.choice(responses)
def extract_name_from_message(message: str) -> Optional[str]:
    """Extract user's name from introduction message with enhanced validation"""
    import re
    message_lower = message.lower().strip()
    name = None
    
    # Skip if it's dissatisfaction
    if is_dissatisfaction(message):
        return None
    
    # Comprehensive blacklists for name extraction
    excluded_words = {
        # Prepositions
        'on', 'at', 'in', 'to', 'for', 'with', 'by', 'from', 'about', 'into', 
        'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 
        'off', 'over', 'under', 'again', 'further', 'then', 'once',
        # Articles and common words
        'the', 'a', 'an', 'and', 'or', 'but', 'so', 'yet', 'for', 'nor',
        # Negative words
        'not', 'no', 'never', 'neither', 'none', 'nothing', 'nobody',
        # Satisfaction/emotion words - expanded list with common misspellings
        'satisfied', 'happy', 'good', 'bad', 'wrong', 'right', 'sad', 'angry',
        'frustrated', 'frustated', 'frustate', 'disappointed', 'pleased', 'excited', 'worried',
        'mad', 'upset', 'annoyed', 'irritated', 'confused', 'lost', 'tired',
        # Common verbs (expanded)
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'can', 'shall', 'go', 'come', 'get', 'make', 'take',
        'give', 'see', 'know', 'think', 'want', 'need', 'like', 'love', 'hate',
        'find', 'look', 'search', 'seek', 'ask', 'tell', 'say', 'speak', 'talk',
        'listen', 'hear', 'read', 'write', 'work', 'play', 'run', 'walk', 'sit',
        'stand', 'eat', 'drink', 'sleep', 'wake', 'buy', 'sell', 'help', 'try',
        'use', 'open', 'close', 'start', 'stop', 'begin', 'end', 'finish',
        'looking', 'searching', 'finding', 'seeking', 'asking', 'telling',
        'wanting', 'needing', 'trying', 'doing', 'going', 'coming', 'being',
        'having', 'getting', 'making', 'taking', 'giving', 'seeing', 'knowing',
        'thinking', 'working', 'playing', 'running', 'walking', 'sitting',
        'standing', 'eating', 'drinking', 'sleeping', 'waking', 'buying',
        'selling', 'helping', 'using', 'opening', 'closing', 'starting',
        'stopping', 'beginning', 'ending', 'finishing',
        # Pronouns
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
        # Numbers (common ones)
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'zero', 'first', 'second', 'third', 'last',
        # Other common words
        'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
        'what', 'who', 'which', 'whom', 'whose',
        # Common nouns that are not names
        'website', 'chatbot', 'bot', 'system', 'service', 'company', 'business',
        'project', 'work', 'job', 'task', 'problem', 'issue', 'question', 'answer',
        'solution', 'help', 'assistance', 'support', 'information', 'data',
        'time', 'day', 'night', 'morning', 'evening', 'year', 'month', 'week',
        'place', 'location', 'area', 'city', 'country', 'world', 'earth',
        'thing', 'stuff', 'item', 'object', 'product', 'service', 'tool',
        'book', 'movie', 'music', 'food', 'water', 'money', 'price', 'cost'
    }
    
    def is_phone_number(text):
        """Check if text is a phone number"""
        # Remove common separators and spaces
        clean_text = re.sub(r'[\s\-\(\)\+]', '', text)
        # Check if it's all digits and reasonable length for phone number
        if clean_text.isdigit() and 7 <= len(clean_text) <= 15:
            return True
        # Check for specific patterns
        phone_patterns = [
            r'\b\d{10}\b',  # 10 digits
            r'\b\d{3}-\d{3}-\d{4}\b',  # 123-456-7890
            r'\b\+?\d{1,3}\s?\d{10}\b',  # +91 1234567890
            r'\b\d{5}\s?\d{5}\b'  # 12345 67890
        ]
        for pattern in phone_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def contains_phone_number(text):
        """Check if text contains phone number pattern"""
        return is_phone_number(text)
    
    def has_context_words(text):
        """Check if text contains context words that indicate it's not a name introduction"""
        context_indicators = [
            'looking', 'searching', 'finding', 'seeking', 'asking', 'telling',
            'wanting', 'needing', 'trying', 'doing', 'going', 'coming',
            'for', 'about', 'regarding', 'concerning', 'website', 'chatbot',
            'bot', 'system', 'service', 'company', 'business', 'project',
            'work', 'job', 'task', 'problem', 'issue', 'question', 'answer',
            'solution', 'help', 'assistance', 'support', 'information'
        ]
        words = text.lower().split()
        return any(indicator in words for indicator in context_indicators)
    
    def is_valid_name(word):
        """Enhanced name validation with multiple layers"""
        if not word or len(word) < 2:
            return False
        
        # Remove punctuation
        word = word.strip('.,!?;:')
        
        # Must be alphabetic only
        if not word.isalpha():
            return False
        
        # Length check: 2-20 characters
        if len(word) < 2 or len(word) > 20:
            return False
        
        # Must not be in excluded words blacklist
        if word.lower() in excluded_words:
            return False
        
        # Must not be a phone number
        if is_phone_number(word):
            return False
        
        # Must not start with common non-name prefixes
        if word.lower().startswith(('mr', 'mrs', 'ms', 'dr', 'prof')):
            return False
    
        # Capitalization check: proper names should be properly capitalized
        # Allow: "John", "Ravi", "Sarah"
        # Reject: "JOHN" (all caps), "john" (all lowercase if > 3 chars)
        if word.isupper() and len(word) > 1:
            return False
        if word.islower() and len(word) > 10:  # Very long lowercase words are likely not names
            return False
        
        # Additional check: reject if it looks like a verb/noun
        if len(word) > 6 and word.lower() in ['looking', 'searching', 'finding', 'seeking', 'asking', 'telling']:
            return False
        
        return True
    
    # Context analysis - reject if sentence contains context words
    if has_context_words(message_lower):
        return None
    
    # Try to extract name from very specific patterns only
    if 'my name is' in message_lower:
        parts = message_lower.split('my name is')
        if len(parts) > 1:
            remaining = parts[1].strip()
            # Get first word only and validate
            words = remaining.split()
            if words:
                potential_name = words[0]
                if is_valid_name(potential_name):
                    name = potential_name.capitalize()
    
    elif 'i am' in message_lower and not contains_phone_number(message):
        # Only extract if "i am" is followed by a single word and it's the end of meaningful content
        parts = message_lower.split()
        for i, part in enumerate(parts):
            if part == 'am' and i + 1 < len(parts):
                potential_name = parts[i + 1]
                # Check if this is likely the end of the name introduction
                remaining_words = parts[i + 2:] if i + 2 < len(parts) else []
                # If there are more words after potential name, they should not be context words
                if remaining_words and has_context_words(' '.join(remaining_words)):
                    continue
                if is_valid_name(potential_name):
                    name = potential_name.capitalize()
                break
    
    elif 'call me' in message_lower:
        # Check if it's a phone number request first
        if contains_phone_number(message):
            return None
        parts = message_lower.split('call me')
        if len(parts) > 1:
            remaining = parts[1].strip()
            # Check if remaining text contains "on" followed by numbers (phone pattern)
            if ' on ' in remaining:
                return None
            words = remaining.split()
            for word in words:
                if is_valid_name(word):
                    name = word.capitalize()
                    break
    
    elif 'mera naam' in message_lower:
        parts = message_lower.split('mera naam')
        if len(parts) > 1:
            potential_name = parts[1].strip().split()[0]
            if is_valid_name(potential_name):
                name = potential_name.capitalize()
    
    elif 'main' in message_lower and 'hun' in message_lower:
        # Handle "main [name] hun" pattern
        parts = message_lower.split()
        try:
            main_index = parts.index('main')
            hun_index = parts.index('hun')
            if hun_index == main_index + 2:  # name is between 'main' and 'hun'
                potential_name = parts[main_index + 1]
                if is_valid_name(potential_name):
                    name = potential_name.capitalize()
        except (ValueError, IndexError):
            pass
    
    # Final validation - ensure name is not a common word
    if name and is_valid_name(name):
        return name
    return None

def is_bot_identity_question(message: str) -> bool:
    """Check if message is asking about bot's identity/name"""
    message_lower = message.lower().strip()
    patterns = [
        'what is your name', 'tell me your name', 'who are you',
        'what do you call yourself', 'your name', 'what\'s your name',
        'apka naam kya hai', 'tumhara naam kya hai', 'aap kaun hain'
    ]
    return any(pattern in message_lower for pattern in patterns)

def get_bot_identity_response(language: str = 'english') -> str:
    """Generate response when user asks about bot's identity"""
    import random
    
    if language == 'hindi':
        responses = [
            "Main aapka AI assistant hun, IT solutions ke liye.",
            "Mera naam AI assistant hai. Main aapka AI assistant hun.",
            "Main aapka AI assistant hun, jo aapki madad ke liye yahan hun.",
            "Namaste! Main aapka AI assistant hun, aapki IT services ke liye.",
            "Main aapka AI chatbot hun, jo aapki madad ke liye yahan hun.",
            "Main aapka AI assistant hun - cloud computing, ERP, CRM solutions ke expert.",
            "Main aapka AI assistant hun, jo aapko IT solutions deta hun."
        ]
    else:
        responses = [
            "I'm your AI assistant, here to help you with IT solutions.",
            "I'm your AI assistant. I'm here to help you with IT services.",
            "I'm your AI assistant, an AI-powered chatbot created to help you.",
            "Hello! I'm your AI assistant, your friendly helper for IT solutions.",
            "I'm your AI assistant - I specialize in cloud computing, ERP, and CRM solutions.",
            "Hi there! I'm your AI assistant, here to assist you with IT services.",
            "I'm your AI assistant, designed to help with IT offerings.",
            "Greetings! I'm your AI assistant, your go-to AI for IT solutions and services.",
            "I'm your AI assistant, created to help with your tech needs.",
            "Hello! I'm your AI assistant, your companion for all IT-related queries."
        ]
    
    return random.choice(responses)

def get_name_recall_response(stored_name: Optional[str], user_language: str = 'english') -> str:
    """Generate response when user asks about their name"""
    import random
    
    if user_language == 'hindi':
        if stored_name:
            responses = [
                f"Haan, aapka naam {stored_name} hai! Main yaad rakhta hun. Aap kya janna chahte hain Fasc Ai ke bare mein?",
                f"Bilkul yaad hai! Aap {stored_name} hain. Main yahan hun aapki IT solutions ke bare mein madad karne ke liye. Kya puchhna chahte hain?",
                f"Ji haan, aapne mujhe bataya tha aapka naam {stored_name} hai. Main aapki kya madad kar sakta hun aaj?"
            ]
        else:
            responses = [
                "Maafi chahta hun, aapne abhi tak mujhe apna naam nahi bataya. Aap mujhe bata sakte hain? Main yaad rakhna chahunga!",
                "Main nahi jaanta aapka naam. Kya aap mujhe bata sakte hain? Main aapka AI assistant hun aur aapki madad karna chahta hun.",
                "Aapne mujhe apna naam nahi bataya hai. Kya aap share kar sakte hain? Main yaad rakhunga!"
            ]
    else:
        if stored_name:
            responses = [
                f"Yes, your name is {stored_name}! I remember. I'm here to help you with Fasc Ai's IT solutions.",
                f"Of course I remember! You're {stored_name}. I'm here to help you with our IT solutions.",
                f"Yes, you told me your name is {stored_name}. I'm here to assist you with Fasc Ai's services."
            ]
        else:
            responses = [
                "I'm sorry, you haven't told me your name yet. Would you like to share it? I'd love to remember it!",
                "I don't know your name yet. Could you tell me? I'm your AI assistant and I'd like to help you.",
                "You haven't shared your name with me. Would you like to? I'll remember it!"
            ]
    
    return random.choice(responses)
def get_personal_introduction_response(message: str, user_language: str = 'english') -> str:
    """Generate response for personal introductions"""
    import random
    
    # Extract name using helper function
    name = extract_name_from_message(message)
    
    if user_language == 'hindi':
        if name:
            responses = [
                f"Namaste {name}! Aapse mil kar khushi hui. Main aapka AI assistant hun aur aapki IT solutions ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                f"Hello {name}! Aapka swagat hai. Main yahan hun aapki cloud computing, ERP, CRM, AI solutions ke bare mein jaankari dene ke liye. Kya puchhna chahte hain?",
                f"Hi {name}! Main aapka friendly AI assistant hun. Fasc Ai ke services ke bare mein koi sawal hai?"
            ]
        else:
            responses = [
                "Namaste! Aapse mil kar achha laga. Main aapka AI assistant hun aur aapki IT solutions ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                "Hello! Aapka swagat hai. Main yahan hun aapki cloud computing, ERP, CRM, AI solutions ke bare mein jaankari dene ke liye. Kya puchhna chahte hain?",
                "Hi! Main aapka friendly AI assistant hun. Fasc Ai ke services ke bare mein koi sawal hai?"
            ]
    else:
        if name:
            responses = [
                f"Nice to meet you, {name}! I'm your AI assistant and I'm here to help you with IT solutions.",
                f"Hello {name}! Great to meet you. I can help you learn about our cloud computing, ERP, CRM, and AI solutions.",
                f"Hi {name}! I'm your friendly AI assistant. I'm here to help you with Fasc Ai's services."
            ]
        else:
            responses = [
                "Nice to meet you! I'm your AI assistant and I'm here to help you with IT solutions.",
                "Hello! Great to meet you. I can help you learn about our cloud computing, ERP, CRM, and AI solutions.",
                "Hi! I'm your friendly AI assistant. I'm here to help you with Fasc Ai's services."
            ]
    
    return random.choice(responses)
def get_personality_response(message: str, user_language: str = 'english') -> str:
    """Generate warm, friendly response for personality questions"""
    message_lower = message.lower().strip()
    
    if user_language == 'hindi':
        # Hindi personality responses
        if any(word in message_lower for word in ['dost', 'friendly', 'achhe', 'helpful']):
            responses = [
                "Haan bilkul! Main aapka friendly assistant hun. Main yahan hun aapki help ke liye Fasc Ai ke services ke bare me. Aap kya janna chahte hain?",
                "Zaroor! Main aapke saath friendly way me baat karne ke liye ready hun. Fasc Ai ke IT solutions ke bare me koi sawal hai?",
                "Bilkul friendly hun! Main aapki har possible help kar sakta hun Fasc Ai ke services ke bare me. Kya puchhna chahte hain?",
                "Haan main dost ban sakta hun! Main yahan hun aapki madad ke liye Fasc Ai ke IT solutions ke bare mein. Aaj main aapki kya madad kar sakta hun?",
                "Bilkul! Main aapka dost ban kar aapki madad karna chahunga. Mera kaam Fasc Ai ki IT solutions ke bare mein jaankari dena hai. Aaj main aapki kya madad kar sakta hun?"
            ]
        else:
            responses = [
                "Haan bilkul! Main aapka friendly assistant hun. Main yahan hun aapki help ke liye Fasc Ai ke services ke bare me. Aap kya janna chahte hain?",
                "Zaroor! Main aapke saath friendly way me baat karne ke liye ready hun. Fasc Ai ke IT solutions ke bare me koi sawal hai?",
                "Bilkul friendly hun! Main aapki har possible help kar sakta hun Fasc Ai ke services ke bare me. Kya puchhna chahte hain?"
            ]
    else:
        # English personality responses
        if any(word in message_lower for word in ['friend', 'dost']):
            responses = [
                "I'd be happy to help you as a friendly AI assistant! My purpose is to provide information about Fasc Ai's IT solutions.",
                "As an AI, I don't have personal feelings, but I'm designed to be very helpful and friendly! I'm here to assist you with Fasc Ai's services.",
                "I'm designed to be your friendly AI assistant! I'm here to help you learn about Fasc Ai's IT solutions and services."
            ]
        else:
            responses = [
                "Yes, I try to be friendly and helpful! I'm here to assist you with Fasc Ai's services in a warm, supportive way.",
                "Absolutely! I'm designed to be your friendly AI assistant. I'd be happy to help you learn about our IT solutions and services.",
                "Of course! I'm here to be your helpful, friendly guide to Fasc Ai's offerings.",
                "Yes, I aim to be friendly and approachable! I'm excited to help you discover what Fasc Ai can do for your business."
            ]
    
    import random
    return random.choice(responses)

def get_emotional_response(message: str, user_language: str = "en") -> str:
    """Generate appropriate response for emotional expressions"""
    message_lower = message.lower().strip()
    
    if any(word in message_lower for word in ['happy', 'excited', 'pleased', 'good', 'great', 'wonderful', 'amazing', 'fantastic']):
        return "I'm glad to hear that! I'm here to help with questions about our IT solutions, cloud computing, ERP, CRM, or AI services."
    
    elif any(word in message_lower for word in ['sad', 'frustrated', 'angry', 'upset', 'annoyed', 'disappointed', 'worried']):
        return "I'm sorry to hear that. I'm here to help you with our IT solutions. Let me know what you need."
    
    elif any(word in message_lower for word in ['bad', 'terrible', 'awful']):
        return "I understand your concerns. I'm committed to providing excellent service. Let me know what specific issues you're facing."
    
    else:
        return "I understand how you're feeling. I'm here to help with questions about our IT solutions, cloud computing, ERP, CRM, or AI services."

def get_user_doubt_response(message: str, user_language: str = "en") -> str:
    """Generate empathetic response for user doubt scenarios"""
    message_lower = message.lower().strip()
    
    # Check for specific doubt patterns and provide tailored responses
    if any(word in message_lower for word in ['thank you', 'thanks']):
        return "I understand your hesitation. I'm designed to help with IT solutions, cloud computing, ERP, CRM, and AI implementations. I've helped many clients with similar concerns."
    
    elif any(word in message_lower for word in ['not helpful', 'not useful', 'can\'t do it']):
        return "I understand your frustration. I'm constantly learning and improving. Our team at Fasc Ai specializes in IT solutions, cloud computing, ERP, CRM, and AI implementations."
    
    elif any(word in message_lower for word in ['don\'t know', 'won\'t understand', 'not smart enough']):
        return "I understand your concern. While I may not know everything, I'm designed to help with IT solutions, cloud computing, ERP, CRM, and AI implementations. I can access our company's knowledge base and connect you with our expert team when needed."
    
    else:
        return "I understand your hesitation. I'm designed to help with IT solutions, cloud computing, ERP, CRM, and AI implementations. I've helped many clients with similar concerns."

def get_help_response(message: str, user_language: str = "en") -> Optional[str]:
    """Generate appropriate response for help requests - simplified to let RAG handle most cases"""
    message_lower = message.lower().strip()
    
    # Check for minimal negative patterns - if found, return None to let RAG handle
    negative_patterns = [
        "don't need", "dont need", "do not need",
        "don't want", "dont want", "do not want",
        "i don't need", "i dont need", "i do not need",
        "i am not your client", "i am not your customer", "i'm not your client"
    ]
    
    if any(pattern in message_lower for pattern in negative_patterns):
        return None  # Let RAG flow handle negative cases with human-like responses
    
    # For all other help requests, return None to let RAG handle
    # RAG will provide better context-aware, human-like, and accurate responses
    return None

def is_off_topic(message: str) -> bool:
    """Detect if a query is off-topic or unrelated to Fasc Ai business"""
    return get_off_topic_category(message) is not None

def get_off_topic_response(message: str) -> str:
    """Generate human-like, categorized response for off-topic queries"""
    import random
    
    category = get_off_topic_category(message)
    
    responses = {
        'abusive': [
            "I appreciate you reaching out, but let's keep our conversation professional. I'm here to help you learn about Fasc Ai's IT solutions.",
            
            "I understand you might be frustrated, but I'm here to assist you professionally. I can help you with Fasc Ai's cloud transformation, ERP, CRM, or other IT solutions.",
            
            "Let's focus on how Fasc Ai can help your business. I'm here to discuss our technology solutions in a respectful manner."
        ],
        
        'other_company': [
            "I specialize in Fasc Ai's solutions rather than other companies. However, I'd be happy to tell you how we compare! We've successfully delivered 250+ projects with cutting-edge IT solutions.",
            
            "That's outside my expertise, but here's what I can tell you - Fasc Ai Ventures Private Limited has worked with major clients like MOF, Max Life, and Lenovo. Our solutions can benefit your business.",
            
            "While I focus on Fasc Ai's services, I can share that our clients often choose us for our personalized approach and proven track record."
        ],
        
        'job_other_company': [
            "I can't help with opportunities at other companies, but I can tell you that Fasc Ai is growing! We're working on exciting projects with major clients.",
            
            "While I don't have insights into other companies' hiring, Fasc Ai Ventures Private Limited is always looking for talented people! We're an innovative IT solutions company with 250+ successful projects. We offer cloud computing, ERP, CRM, AI solutions, and IoT services."
        ],
        
        'unrelated': [
            "That's a bit outside my wheelhouse! My expertise is in Fasc Ai's technology solutions. We offer cloud services, digital transformation, and enterprise applications.",
            
            "I wish I could help with that, but I specialize in IT solutions for businesses! We can help with technology upgrades and digital transformation for your organization.",
            
            "Ha, I'm not the best person for that question! But I'm great at discussing Fasc Ai's services - we offer cloud transformation, ERP, CRM, AI implementations, and more."
            ,
            "That topic is a little outside our scope. If you'd like to talk about cloud transformation, ERP/CRM, AI, or IoT, I'm totally in my comfort zone.",
            "I'm focused on Fasc Ai's digital solutions—cloud, ERP, CRM, AI, and IoT. Let me know if you want insights there, I'm happy to help."
        ],
        
        'general': [
            "I don't have specific information about that, but I can tell you all about Fasc Ai's IT solutions. We've helped businesses transform with cloud services, enterprise applications, and AI.",
            
            "That's not quite in my area, but I'm really good at discussing technology solutions! Fasc Ai Ventures Private Limited specializes in cloud transformation, ERP, CRM, and digital innovation.",
            
            "I'm not able to assist with that particular topic. However, if you're looking for IT solutions, cloud services, or digital transformation expertise, you're in the right place! We offer cloud computing, ERP, CRM, AI solutions, and IoT services."
            ,
            "Let me steer things back to what we do best—cloud, ERP, CRM, AI, and IoT solutions. Ask me anything about those and I’ll gladly dive in.",
            "While I can’t cover that subject, I can absolutely help you explore Fasc Ai’s technology offerings: automation with AI, cloud transformation, ERP/CRM rollouts, and more."
        ]
    }
    
    return random.choice(responses.get(category, responses['general']))

# Greeting detection function
def is_greeting(message: str) -> bool:
    """Check if message is a casual greeting"""
    greeting_words = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'namaste', 'namaskar', 'salam', 'hiya', 'howdy', 'greetings',
        'kaise ho', 'kaise hain', 'kaise hain aap', 'aap kaise hain'
    ]
    
    message_lower = message.lower().strip()
    
    # Check for exact matches
    if message_lower in greeting_words:
        return True
    
    # Check for greetings with additional words (like "good morning sir")
    for greeting in greeting_words:
        if message_lower.startswith(greeting) and len(message_lower.split()) <= 3:
            return True
    
    return False
def get_greeting_response(message: str, user_language: str = 'english') -> str:
    """Generate appropriate casual response for greetings with domain context"""
    message_lower = message.lower().strip()
    import random
    
    if user_language == 'hindi':
        # Hindi greetings with variations
        if any(word in message_lower for word in ['namaste', 'namaskar']):
            responses = [
                "Namaste! Main aapka AI assistant hun. Main aapki IT solutions, cloud services, ERP, CRM, AI implementations ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                "Namaskar! Main aapka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI solutions ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                "Namaste! Main yahan hun aapki IT services ke bare mein madad karne ke liye. Cloud hosting, ERP, CRM, AI solutions - sabke bare mein puchh sakte hain. Aaj main aapki kya madad kar sakta hun?"
            ]
        elif any(word in message_lower for word in ['kaise ho', 'kaise hain', 'aap kaise hain']):
            responses = [
                "Main theek hun! Main aapka AI assistant hun. Main aapki IT solutions, cloud services, aur digital transformation services ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                "Bilkul theek hun! Main aapka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI implementations ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                "Main achha hun! Main yahan hun aapki IT services ke bare mein madad karne ke liye. Cloud hosting, ERP, CRM, AI solutions - sabke bare mein puchh sakte hain."
            ]
        elif 'good morning' in message_lower:
            responses = [
                "Good morning! Main aapka AI assistant hun. Main aapki IT solutions, cloud services, ERP, CRM, AI implementations ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                "Good morning! Main aapka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI solutions ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                "Good morning! Main yahan hun aapki IT services ke bare mein madad karne ke liye. Cloud hosting, ERP, CRM, AI solutions - sabke bare mein puchh sakte hain."
            ]
        elif 'good afternoon' in message_lower:
            responses = [
                "Good afternoon! Main aapka AI assistant hun. Main aapki IT solutions, cloud services, ERP, CRM, AI implementations ke bare mein madad kar sakta hun. Main aapki kya madad kar sakta hun?",
                "Good afternoon! Main aapka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI solutions ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                "Good afternoon! Main yahan hun aapki IT services ke bare mein madad karne ke liye. Cloud hosting, ERP, CRM, AI solutions - sabke bare mein puchh sakte hain."
            ]
        elif 'good evening' in message_lower:
            responses = [
                "Good evening! Main aapka AI assistant hun. Main aapki IT solutions, cloud services, ERP, CRM, AI implementations ke bare mein madad kar sakta hun. Main aapki kya madad kar sakta hun?",
                "Good evening! Main aapka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI solutions ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                "Good evening! Main yahan hun aapki IT services ke bare mein madad karne ke liye. Cloud hosting, ERP, CRM, AI solutions - sabke bare mein puchh sakte hain."
            ]
        elif 'hi' in message_lower:
            responses = [
                "Hi there! Main apka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI implementations ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                "Hi there! Main apka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI implementations ke bare mein madad kar sakta hun. Kya puchhna chahte hain?",
                "Hi there! Main apka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI implementations ke bare mein jaankari de sakta hun. Aap kya janna chahte hain?"
            ]
        elif 'hello' in message_lower:
            responses = [
                "Hello! Main aapka AI assistant hun, yahan aapki IT solutions, cloud transformation, ERP, CRM, AI services ke bare mein jaankari dene ke liye. Aapko kya pasand hai?",
                "Hello! Main aapka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI solutions ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                "Hello! Main yahan hun aapki IT services ke bare mein madad karne ke liye. Cloud hosting, ERP, CRM, AI solutions - sabke bare mein puchh sakte hain."
            ]
        elif any(word in message_lower for word in ['hey']):
            responses = [
                "Hey! Main aapka AI assistant hun. Aap mere se IT solutions, cloud services, ya koi bhi digital transformation services ke bare mein puchh sakte hain. Main aapki kya madad kar sakta hun?",
                "Hey there! Main aapka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI solutions ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                "Hey! Main yahan hun aapki IT services ke bare mein madad karne ke liye. Cloud hosting, ERP, CRM, AI solutions - sabke bare mein puchh sakte hain."
            ]
        else:
            responses = [
                "Hi! Main aapka AI assistant hun. Main aapki IT solutions aur services ke bare mein madad kar sakta hun. Aap kya janna chahte hain?",
                "Hello! Main aapka AI assistant hun. Main aapki cloud computing, ERP systems, aur AI solutions ke bare mein jaankari de sakta hun. Kya puchhna chahte hain?",
                "Hi there! Main yahan hun aapki IT services ke bare mein madad karne ke liye. Cloud hosting, ERP, CRM, AI solutions - sabke bare mein puchh sakte hain."
            ]
    else:
        # English greetings with variations
        if any(word in message_lower for word in ['namaste', 'namaskar']):
            responses = [
                "Namaste! I'm your AI assistant. I can help you with our IT solutions, cloud services, ERP, CRM, AI implementations, and more.",
                "Namaskar! I'm your AI assistant. I can provide information about cloud computing, ERP systems, and AI solutions.",
                "Namaste! I'm here to help you with our IT services. Cloud hosting, ERP, CRM, AI solutions - I'm ready to assist you."
            ]
        elif any(word in message_lower for word in ['kaise ho', 'kaise hain', 'aap kaise hain']):
            responses = [
                "I'm doing great! I'm your AI assistant. I can help you learn about our IT solutions, cloud services, and digital transformation services.",
                "I'm excellent! I'm your AI assistant. I can provide information about cloud computing, ERP systems, and AI implementations.",
                "I'm wonderful! I'm here to help you with our IT services. Cloud hosting, ERP, CRM, AI solutions - I'm ready to assist you."
            ]
        elif 'good morning' in message_lower:
            responses = [
                "Good morning! I'm your AI assistant. I can help you with our IT solutions, cloud services, ERP, CRM, AI implementations, and more.",
                "Good morning! I'm your AI assistant. I can provide information about cloud computing, ERP systems, and AI solutions.",
                "Good morning! I'm here to help you with our IT services. Cloud hosting, ERP, CRM, AI solutions - I'm ready to assist you."
            ]
        elif 'good afternoon' in message_lower:
            responses = [
                "Good afternoon! I'm your AI assistant. I can help you with our IT solutions, cloud services, ERP, CRM, AI implementations, and more.",
                "Good afternoon! I'm your AI assistant. I can provide information about cloud computing, ERP systems, and AI solutions.",
                "Good afternoon! I'm here to help you with our IT services. Cloud hosting, ERP, CRM, AI solutions - I'm ready to assist you."
            ]
        elif 'good evening' in message_lower:
            responses = [
                "Good evening! I'm your AI assistant. I can help you with our IT solutions, cloud services, ERP, CRM, AI implementations, and more.",
                "Good evening! I'm your AI assistant. I can provide information about cloud computing, ERP systems, and AI solutions.",
                "Good evening! I'm here to help you with our IT services. Cloud hosting, ERP, CRM, AI solutions - I'm ready to assist you."
            ]
        elif 'hi' in message_lower:
            responses = [
                "Hi there! I'm your friendly AI assistant. I can provide information about our cloud computing, ERP systems, and AI implementations.",
                "Hi there! I'm your friendly AI assistant. I can help you learn about our cloud computing, ERP systems, and AI implementations.",
                "Hi there! I'm your friendly AI assistant. I can assist you with information about our cloud computing, ERP systems, and AI implementations."
            ]
        elif 'hello' in message_lower:
            responses = [
                "Hello! I'm your AI assistant, here to help you with information about our IT solutions, cloud transformation, ERP, CRM, AI services, and more.",
                "Hello! I'm your AI assistant. I can provide information about cloud computing, ERP systems, and AI solutions.",
                "Hello! I'm here to help you with our IT services. Cloud hosting, ERP, CRM, AI solutions - I'm ready to assist you."
            ]
        elif any(word in message_lower for word in ['hey']):
            responses = [
                "Hey there! I'm your AI assistant. I can help you with our IT solutions, cloud services, or any of our digital transformation offerings.",
                "Hey! I'm your AI assistant. I can provide information about cloud computing, ERP systems, and AI solutions.",
                "Hey there! I'm here to help you with our IT services. Cloud hosting, ERP, CRM, AI solutions - I'm ready to assist you."
            ]
        else:
            responses = [
                "Hi there! I'm your AI assistant. I can help you with our IT solutions and services.",
                "Hello! I'm your AI assistant. I can provide information about cloud computing, ERP systems, and AI solutions.",
                "Hi! I'm here to help you with our IT services. Cloud hosting, ERP, CRM, AI solutions - I'm ready to assist you."
            ]
    
    return random.choice(responses)

# Service detection function
def detect_specific_service(message: str) -> Optional[str]:
    """Detect which specific service user is asking about"""
    message_lower = message.lower()
    
    service_keywords = {
        'ai': ['ai', 'artificial intelligence', 'machine learning', 'ml', 'chatbot', 'ai services', 'ai solutions'],
        'crm': ['crm', 'customer relationship management', 'crm services', 'crm solutions'],
        'erp': ['erp', 'enterprise resource planning', 'erp services', 'erp systems'],
        'cloud': ['cloud', 'cloud computing', 'cloud hosting', 'cloud services'],
        'iot': ['iot', 'internet of things'],
        'website': ['website', 'web development', 'web design', 'ecommerce', 'e-commerce']
    }
    
    # Check which service matches (prioritize specific matches)
    for service, keywords in service_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            return service
    
    return None  # Generic service inquiry
# Service inquiry detection
def is_service_inquiry(message: str) -> bool:
    """Check if message is asking about services"""
    message_lower = message.lower().strip()
    
    # Service inquiry patterns
    service_patterns = [
        'tell me about your services', 'what services do you offer', 'your services',
        'about your services', 'what services', 'services you provide', 'your offerings',
        'what do you offer', 'services offered', 'your service', 'about services',
        'what are your services', 'list your services', 'services available',
        'what services are available', 'services you have', 'your service offerings',
        'services provided', 'what services do you have', 'services offered by',
        'tell me about services', 'about your service', 'what service do you offer',
        'services you offer', 'your service offerings', 'what are the services',
        'services list', 'available services', 'services you provide'
    ]
    
    # Check for service inquiry patterns
    for pattern in service_patterns:
        if pattern in message_lower:
            return True
    
    # Check for combination of keywords
    service_keywords = ['services', 'service', 'offerings', 'offer', 'provide']
    inquiry_keywords = ['tell', 'what', 'about', 'list', 'show', 'describe']
    
    has_service_keyword = any(keyword in message_lower for keyword in service_keywords)
    has_inquiry_keyword = any(keyword in message_lower for keyword in inquiry_keywords)
    
    if has_service_keyword and has_inquiry_keyword:
        return True
    
    return False
# Acknowledgment detection
def is_acknowledgment(message: str) -> bool:
    """Check if message is acknowledgment/thanks - more specific to avoid catching service inquiries"""
    message_lower = message.lower().strip()
    
    # Pure acknowledgment words (single words or short phrases)
    pure_acknowledgments = [
        'thanks', 'thank you', 'thank', 'thx', 'thanx', 'appreciate',
        'ok', 'okay', 'got it', 'understood', 'clear', 'perfect', 'great',
        'nice', 'good', 'cool', 'awesome', 'makes sense', 'that helps',
        'theek hai', 'accha', 'samajh gaya', 'dhanyawad', 'shukriya',
        'bilkul', 'zaroor', 'sahi hai', 'badhiya'
    ]
    
    # Check for exact matches or very short phrases
    for word in pure_acknowledgments:
        if word == message_lower:  # Exact match
            return True
        # Only match if it's a short phrase (max 3 words) and doesn't contain service keywords
        elif (word in message_lower and 
              len(message_lower.split()) <= 3 and 
              not any(service_word in message_lower for service_word in ['service', 'services', 'tell', 'about', 'what', 'offer'])):
            return True
    
    return False

def get_acknowledgment_response(user_language: str = 'english') -> str:
    """Generate appropriate response for acknowledgments"""
    import random
    
    if user_language == 'hindi':
        responses = [
            "Koi baat nahi! Fasc Ai ke IT solutions ke bare me aur kuch puchh sakte hain.",
            "Khushi hui madad kar ke! Agar aur koi sawal ho Fasc Ai ke services ke bare me to puchhiye.",
            "Aapka swagat hai! Fasc Ai ke IT solutions ke bare mein aur jaankari chahiye to puchhiye.",
            "Bilkul! Fasc Ai ke services ke bare mein aur koi sawal ho to zaroor puchhiye.",
            "Dhanyawad! Fasc Ai ke IT solutions ke bare mein aur madad chahiye to main yahan hun."
        ]
    else:
        responses = [
            "You're welcome! Feel free to ask if you need anything else about Fasc Ai's IT solutions.",
            "Glad I could help! Let me know if you have more questions about our services.",
            "Happy to help! Don't hesitate to reach out if you need more information about Fasc Ai.",
            "My pleasure! If you have more questions about Fasc Ai's services, I'm here to help.",
            "You're very welcome! Feel free to ask about any other Fasc Ai IT solutions."
        ]
    
    return random.choice(responses)

# Goodbye detection
def is_goodbye(message: str) -> bool:
    """Check if message is goodbye/end chat"""
    message_lower = message.lower().strip()
    goodbye_words = [
        'bye', 'goodbye', 'good bye', 'see you', 'later', 'end chat',
        'that\'s all', 'done', 'exit', 'quit', 'close',
        'namaste', 'alvida', 'phir milte hain', 'chaliye', 'bye bye',
        'kal milte hain', 'kal milte hai', 'kal baat karunga', 'kal baat karenge',
        'phir baat karte hain', 'phir baat karenge', 'baad me baat karte hain'
    ]
    
    for word in goodbye_words:
        if word in message_lower and len(message_lower.split()) <= 4:
            return True
    
    return False

def get_goodbye_response(user_language: str = 'english') -> str:
    """Generate appropriate goodbye response"""
    import random
    
    if user_language == 'hindi':
        responses = [
            "Dhanyawad Fasc Ai se baat karne ke liye! Agar future me aur sawal hain to main yahan hun. fascai.com visit kariye.",
            "Aapke saath baat kar ke achha laga! Fasc Ai ke IT solutions ke bare me aur sawal ho to zaroor puchhiye. Achha din guzare!",
            "Shukriya Fasc Ai se baat karne ke liye! Agar aur koi sawal ho to main yahan hun. fascai.com par jaankari le sakte hain.",
            "Aapka dhanyawad! Fasc Ai ke IT solutions ke bare mein aur jaankari chahiye to zaroor puchhiye. Achha din!",
            "Khushi hui Fasc Ai se baat kar ke! Agar future mein aur sawal hain to main yahan hun. fascai.com visit kariye."
        ]
    else:
        responses = [
            "Thank you for chatting with Fasc Ai! If you have more questions in the future, I'm here to help. Visit fascai.com for more information.",
            "It was great talking with you! Feel free to return anytime with questions about Fasc Ai's IT solutions. Have a great day!",
            "Thanks for reaching out to Fasc Ai! Don't hesitate to come back if you need anything. Visit fascai.com to explore our services.",
            "My pleasure helping you! If you have more questions about Fasc Ai's IT solutions, I'm always here. Have a wonderful day!",
            "Thank you for choosing Fasc Ai! Feel free to return anytime for more information about our services. Visit fascai.com!"
        ]
    
    return random.choice(responses)

# General help request detection
def is_general_help_request(message: str) -> bool:
    """Check if message is requesting general help (not asking about AI capabilities)"""
    message_lower = message.lower().strip()
    
    # FIRST: Check for negative patterns - if user says they DON'T need help, return False
    # This allows RAG flow to handle it and generate human-like acknowledgment
    negative_patterns = [
        "don't need", "dont need", "do not need", "don't want", "dont want", "do not want",
        "not need", "not want", "no need", "never need", "never want",
        "don't need help", "dont need help", "do not need help",
        "don't want help", "dont want help", "do not want help",
        "not need help", "not want help", "no need help",
        "i don't need", "i dont need", "i do not need",
        "i don't want", "i dont want", "i do not want",
        "i don't need your help", "i dont need your help", "i do not need your help",
        "i don't want your help", "i dont want your help", "i do not want your help"
    ]
    
    # If negative pattern matches, return False (let RAG flow handle it)
    for pattern in negative_patterns:
        if pattern in message_lower:
            return False
    
    # FIRST: Check for service-specific keywords - if present, this is NOT a general help request
    # These technical/service questions should go to RAG for intelligent, context-based answers
    service_keywords = [
        'erp', 'crm', 'cloud', 'hosting', 'iot', 'ai', 'artificial intelligence',
        'service', 'services', 'solution', 'solutions', 'system', 'systems',
        'application', 'applications', 'website', 'websites', 'software',
        'set up', 'setup', 'implement', 'implementation', 'install', 'installation',
        'develop', 'development', 'create', 'creation', 'build', 'building',
        'transform', 'transformation', 'migrate', 'migration', 'deploy', 'deployment',
        'automation', 'automate', 'digital transformation', 'business process', 'integration',
        'workflow', 'process automation', 'digitalization', 'digitization', 'api', 'apis'
    ]
    if any(keyword in message_lower for keyword in service_keywords):
        return False  # Service-specific help, let RAG/Groq handle it
    
    # Check for cybersecurity queries FIRST - these should be handled by project_manager
    cybersecurity_patterns = [
        'cybersecurity', 'cyber security', 'security', 'penetration testing',
        'vulnerability assessment', 'security audit', 'security testing',
        'ethical hacking', 'security consulting', 'security services',
        'need help with cybersecurity', 'help with cybersecurity', 'want help with cybersecurity',
        'can you help with cybersecurity', 'assistance with cybersecurity'
    ]

    # If it's a cybersecurity query, return False (let project_manager handle it)
    for pattern in cybersecurity_patterns:
        if pattern in message_lower:
            return False

    # Check for database design queries - these should be handled by project_manager
    database_patterns = [
        'database design', 'database designing', 'db design', 'database architecture',
        'need help with database design', 'help with database design', 'want help with database design',
        'can you help with database design', 'assistance with database design',
        'database help', 'db help', 'database services', 'db services',
        'i need help with database design', 'i need help with db design',
        'help me with database design', 'help me with db design',
        'i need help with database', 'i need help with db'
    ]

    # If it's a database design query, return False (let project_manager handle it)
    for pattern in database_patterns:
        if pattern in message_lower:
            return False
    
    # First check for service-specific help requests - these should NOT be general help
    # BUT exclude database design patterns from this check
    service_specific_patterns = [
        'can you help with', 'help with', 'need help with', 'want help with',
        'assistance with', 'support with', 'guidance with', 'help me with',
        'can you assist with', 'can you support with', 'can you guide with'
    ]
    
    # If it's a service-specific help request, return False (let ChromaDB handle it)
    # BUT skip if it's a database design query (already handled above)
    for pattern in service_specific_patterns:
        if pattern in message_lower:
            # Skip if it's a database design query - check if message contains database-related terms
            if not ('database' in message_lower or 'db ' in message_lower):
                return False
    
    # General help request patterns
    help_patterns = [
        'i need help with something else', 'i want another help', 'i need assistance',
        'help me with', 'i need support', 'i want help', 'i need help',
        'can you help me with', 'help me', 'assist me', 'support me',
        'i need guidance', 'i want assistance', 'i need some help',
        'help me out', 'can you assist', 'i need some assistance',
        'help me please', 'i need help please', 'assistance needed',
        'i need help with', 'help with', 'need help', 'want help'
    ]
    
    # Check for help request patterns
    for pattern in help_patterns:
        if pattern in message_lower:
            return True
    
    # Check for combination of keywords
    help_keywords = ['help', 'assistance', 'support', 'guidance']
    request_keywords = ['need', 'want', 'require', 'looking for', 'seeking']
    
    has_help_keyword = any(keyword in message_lower for keyword in help_keywords)
    has_request_keyword = any(keyword in message_lower for keyword in request_keywords)
    
    if has_help_keyword and has_request_keyword:
        return True
    
    return False

def is_capability_question(message: str) -> bool:
    """Check if message is asking about chatbot/AI capabilities"""
    message_lower = message.lower().strip()
    
    # FIRST: Check for service-specific keywords - if present, this is NOT a generic capability question
    # These technical/service questions should go to RAG for intelligent, context-based answers
    service_keywords = [
        'erp', 'crm', 'cloud', 'hosting', 'iot', 'ai', 'artificial intelligence',
        'service', 'services', 'solution', 'solutions', 'system', 'systems',
        'application', 'applications', 'website', 'websites', 'software',
        'set up', 'setup', 'implement', 'implementation', 'install', 'installation',
        'develop', 'development', 'create', 'creation', 'build', 'building',
        'transform', 'transformation', 'migrate', 'migration', 'deploy', 'deployment'
    ]
    if any(keyword in message_lower for keyword in service_keywords):
        return False  # Service-specific capability question, let RAG/Groq handle it
    
    # Capability question patterns
    capability_patterns = [
        'how can you help me', 'how can you help', 'what can you help me with',
        'what can you do', 'how do you help', 'what do you help with',
        'how do you work', 'what are your capabilities', 'what can you assist with',
        'how do you assist', 'what services do you provide', 'what can you offer',
        'i ask how can you help me', 'tell me how you can help', 'explain how you help',
        'what help can you provide', 'how do you support', 'what support do you offer'
    ]
    
    # Check for capability question patterns
    for pattern in capability_patterns:
        if pattern in message_lower:
            return True
    
    return False

def get_capability_response(user_language: str = 'english', session_id: str = None) -> str:
    """Generate response explaining chatbot capabilities with variety"""
    logger.info(f"DEBUG: get_capability_response called with session_id: {session_id}")
    if user_language == 'hindi':
        responses = [
            "Main aapka AI assistant hun. Main aapki help kar sakta hun website development, ERP systems, CRM solutions, cloud computing, AI implementations, aur IoT services ke bare mein.",
            "Main Fasc AI ka AI assistant hun. Main aapko project management, technical support, aur service information provide kar sakta hun."
        ]
    else:
        responses = [
            "I'm here to help with your IT needs. I can assist with projects, answer questions about our services, or provide technical support.",
            "I can help you with starting new projects, checking existing projects, getting service information, or connecting with our team.",
            "I'm your AI assistant. I can help with website development, ERP systems, CRM solutions, cloud computing, AI implementations, and IoT services.",
            "I assist with project management, technical support, and service information for cloud, ERP, CRM, AI, and IoT solutions.",
            "I can help you with website development, chatbot creation, ERP systems, CRM solutions, cloud computing, and AI implementations."
        ]
    
    # If session_id provided, track capability question count for variety
    if session_id and session_id in conversation_sessions:
        if 'capability_count' not in conversation_sessions[session_id]:
            conversation_sessions[session_id]['capability_count'] = 0
        conversation_sessions[session_id]['capability_count'] += 1
        count = conversation_sessions[session_id]['capability_count']
        selected_response = responses[(count - 1) % len(responses)]
        logger.info(f"Capability response #{count} for session {session_id}: {selected_response[:50]}...")
        return selected_response
    else:
        # Fallback to random selection if no session tracking
        import random
        fallback_response = random.choice(responses)
        logger.info(f"Using fallback capability response (no session tracking): {fallback_response[:50]}...")
        return fallback_response
def get_general_help_response(user_language: str = 'english') -> str:
    """Generate appropriate response for general help requests"""
    import random
    
    if user_language == 'hindi':
        responses = [
            "Main aapki madad ke liye yahan hun. Aapko cloud solutions, ERP, CRM, AI implementations, ya koi aur service ke bare mein batayein.",
            "Bilkul! Main aapki help kar sakta hun IT solutions, cloud computing, ERP, CRM, aur AI services ke bare mein.",
            "Main yahan hun aapki madad ke liye. Aapko kis area mein assistance chahiye?"
        ]
    else:
        responses = [
            "I'm here to help. I can assist with cloud solutions, ERP systems, CRM, AI implementations, and other IT services.",
            "I can help you with IT solutions, cloud computing, ERP, CRM, AI, and other services.",
            "I'm here to assist. What do you need help with?"
        ]
    
    return random.choice(responses)

# Meta/Help detection
def is_meta_question(message: str) -> bool:
    """Check if message is asking about the chatbot itself (more specific)"""
    message_lower = message.lower().strip()
    
    # Remove common profanity to check core question
    cleaned_message = message_lower.replace('hell', '').replace('fuck', '').replace('damn', '').replace('shit', '')
    
    # More specific meta patterns - only questions about AI capabilities/identity
    meta_patterns = [
        'what can you do', 'what do you do', 'who are you', 
        'how does this work', 'what is this', 'your capabilities',
        'who built you', 'who made you', 'what are you',
        'who are u', 'what are u', 'who r u', 'what r u',
        'who the hell are you', 'who the hell', 'who the f*** are you',
        'who the f are you', 'who the hell are u', 'who the f*** are u',
        'how can you help me'  # Only this specific pattern, not general help requests
    ]
    
    # Check both original and cleaned message
    for pattern in meta_patterns:
        if pattern in message_lower or pattern in cleaned_message:
            return True
    
    # Additional check for "who the hell" variations
    if 'who the hell' in message_lower and any(word in message_lower for word in ['are you', 'are u', 'r u']):
        return True
    
    return False

def get_meta_response() -> str:
    """Generate response about chatbot capabilities"""
    import random
    responses = [
        "I'm an AI assistant from Fasc Ai. I help you explore our IT and AI solutions.",
        "I'm your AI assistant. I can answer questions about cloud transformation, ERP, CRM, AI implementations, and IoT solutions.",
        "I'm here to help you explore Fasc Ai's IT solutions. I can discuss our cloud services, ERP, CRM, AI implementations, and past projects."
    ]
    return random.choice(responses)

# Contact info detection
def is_contact_query(message: str) -> bool:
    """Check if message is asking for contact information"""
    message_lower = message.lower().strip()
    contact_patterns = [
        'location', 'address', 'where are you', 'office', 'contact',
        'email', 'phone', 'call', 'telephone', 'reach', 'support hours',
        'how to contact', 'get in touch', 'talk to someone', 'human agent',
        'speak to agent', 'customer service'
    ]
    
    for pattern in contact_patterns:
        if pattern in message_lower:
            return True
    
    return False
def get_contact_response() -> str:
    """Generate response for contact queries"""
    import random
    responses = [
        "You can reach us at info@fascai.com or visit fascai.com/contact for contact details and office locations.",
        "Contact us at info@fascai.com or visit fascai.com/contact for phone numbers and office details.",
        "Email us at info@fascai.com or visit fascai.com/contact for direct assistance."
    ]
    return random.choice(responses)

# Service Information Query Detection
def is_service_info_query(message: str) -> bool:
    """Check if message is asking about services information (not pricing)"""
    message_lower = message.lower().strip()
    
    # Service info patterns - asking ABOUT services, not pricing
    service_info_patterns = [
        'what are your services', 'what services do you offer', 'what services do you provide',
        'tell me about your services', 'what services are available', 'what services do you have',
        'what ai solutions do you have', 'what cloud services do you offer', 'what erp solutions',
        'what crm solutions', 'what services does fasc ai offer', 'what does fasc ai do',
        'what can fasc ai help with', 'what solutions do you provide', 'what do you offer',
        'services you offer', 'services you provide', 'services available'
    ]
    
    for pattern in service_info_patterns:
        if pattern in message_lower:
            return True
    
    # Check for "what" + "services" combination (but not pricing)
    if 'what' in message_lower and 'services' in message_lower:
        # Exclude pricing-related words
        pricing_words = ['cost', 'price', 'pricing', 'expensive', 'cheap', 'how much']
        if not any(word in message_lower for word in pricing_words):
            return True
    
    return False

def get_service_info_response() -> str:
    """Generate response for service information queries"""
    import random
    responses = [
        "We offer cloud computing, ERP, CRM, AI solutions, and IoT services. We've delivered 250+ projects for clients like MOF, Max Life, and Lenovo.",
        "Our services include cloud transformation, ERP systems, CRM platforms, AI implementations, and IoT solutions. We've worked with major clients including MOF, Max Life, Lenovo, Medanta, and Videocon.",
        "We provide cloud computing, ERP, CRM, AI, IoT, and web development services. We've successfully completed 250+ projects across various industries."
    ]
    return random.choice(responses)

# Pricing detection
def is_pricing_query(message: str) -> bool:
    """Check if message is asking about pricing (refined to avoid false positives)"""
    message_lower = message.lower().strip()
    
    # First check if it's a complaint about services - don't treat as pricing
    complaint_words = ['frustrated', 'upset', 'angry', 'not satisfied', 'bad', 'wrong', 'terrible', 'poor']
    if any(word in message_lower for word in complaint_words) and 'services' in message_lower:
        return False
    
    # Exclude non-pricing uses of "rate" keyword
    non_pricing_rate_patterns = [
        'success rate', 'conversion rate', 'error rate', 'performance rate',
        'completion rate', 'satisfaction rate', 'response rate', 'uptime rate'
    ]
    for pattern in non_pricing_rate_patterns:
        if pattern in message_lower:
            return False
    
    # Explicit pricing patterns only (but exclude standalone "rate" without pricing context)
    pricing_patterns = [
        'price', 'pricing', 'cost', 'how much', 'expensive', 'cheap',
        'fees', 'charge', 'payment', 'trial', 'free trial',
        'demo', 'packages', 'plans', 'subscription', 'quote', 'estimate'
    ]
    
    # Check for standalone "rate" with pricing context
    if 'rate' in message_lower:
        pricing_context_words = ['price', 'pricing', 'cost', 'fee', 'charge', 'payment', 'quote']
        if any(word in message_lower for word in pricing_context_words):
            # Has pricing context, treat as pricing
            pass  # Will be caught by pricing_patterns check below
        else:
            # "rate" without pricing context - likely not pricing (could be success rate, etc.)
            return False
    
    for pattern in pricing_patterns:
        if pattern in message_lower:
            return True
    
    # Check for "services" + pricing words combination
    if 'services' in message_lower:
        pricing_context_words = ['cost', 'price', 'pricing', 'how much', 'expensive', 'cheap', 'fees']
        if any(word in message_lower for word in pricing_context_words):
            return True
    
    return False

def get_pricing_response() -> str:
    """Generate response for pricing queries"""
    import random
    responses = [
        "Pricing varies based on your requirements. For a custom quote, email info@fascai.com or visit fascai.com/contact.",
        "We offer flexible pricing based on business needs. Contact info@fascai.com or visit fascai.com/contact for detailed pricing.",
        "For pricing information tailored to your needs, email info@fascai.com or visit fascai.com/contact."
    ]
    return random.choice(responses)

# Policy detection
def is_policy_query(message: str) -> bool:
    """Check if message is asking about company policies"""
    message_lower = message.lower().strip()
    policy_patterns = [
        'policy', 'policies', 'terms', 'conditions', 'terms and conditions',
        'privacy policy', 'refund policy', 'cancellation policy', 'return policy',
        'company policy', 'business policy', 'service policy', 'data policy',
        'terms of service', 'terms of use', 'user agreement', 'legal'
    ]
    
    for pattern in policy_patterns:
        if pattern in message_lower:
            return True
    
    return False

# Frustration detection
def is_frustrated(message: str) -> bool:
    """Check if user seems frustrated or confused"""
    message_lower = message.lower().strip()
    frustration_patterns = [
        'not helping', 'not helpful', 'confused', 'don\'t understand',
        'doesn\'t make sense', 'not clear', 'unclear', 'wrong answer',
        'not working', 'fix this', 'actually help', 'canned repl',
        'stop giving', 'useless', 'what are you saying'
    ]
    
    for pattern in frustration_patterns:
        if pattern in message_lower:
            return True
    
    return False

def get_frustration_response() -> str:
    """Generate empathetic response for frustrated users"""
    import random
    responses = [
        "I apologize if that wasn't clear. I can help you with cloud solutions, ERP, CRM, AI implementations, or our projects.",
        "I'm sorry for the confusion. Let me assist you more directly with Fasc Ai's IT solutions.",
        "My apologies. I can provide better information about our cloud services, ERP, CRM, AI solutions, or other services."
    ]
    return random.choice(responses)

# Complaint/Dissatisfaction detection
def is_dissatisfaction(message: str) -> bool:
    """Check if user is expressing dissatisfaction with the previous response"""
    message_lower = message.lower().strip()
    dissatisfaction_patterns = [
        # Direct dissatisfaction expressions
        'not satisfied', 'not happy', 'not satisfied with', 'unsatisfied',
        'dissatisfied', 'not good', 'not helpful', 'not working',
        'i am not satisfy', 'i am not satisfied', 'i\'m not satisfied',
        'i am not happy', 'i\'m not happy', 'i am not satisfy with',
        'i am not satisfied with', 'i\'m not satisfied with',
        
        # Quality issues
        'that\'s not helpful', 'that doesn\'t help', 'that\'s not what i wanted',
        'that\'s not right', 'that\'s wrong', 'incorrect', 'not what i asked',
        'you didn\'t answer', 'you didn\'t understand', 'that\'s not clear',
        'confusing', 'not useful', 'useless', 'waste of time',
        'not what i need', 'bad answer', 'wrong answer', 'disappointed', 'frustrated',
        'not what i was looking for', 'this doesn\'t help', 'not helpful at all',
        'that doesn\'t work', 'doesn\'t help', 'not correct',
        'not helpful', 'not what i expected', 'annoyed', 'not impressed',
        'this is not what i want', 'this is wrong', 'this is bad',
        
        # Service/product dissatisfaction
        'not satisfied with your', 'not happy with your', 'not good with your',
        'not satisfied with the', 'not happy with the', 'not good with the',
        'dissatisfied with your', 'dissatisfied with the',
        
        # Generic negative expressions
        'this is not good', 'this is not helpful', 'this is not what i wanted',
        'this is wrong', 'this is bad', 'this is useless',
        'it\'s not good', 'it\'s not helpful', 'it\'s not what i wanted',
        'it\'s wrong', 'it\'s bad', 'it\'s useless',
        
        # Hindi/Hinglish patterns
        'accha nahi', 'sahi nahi', 'theek nahi', 'pasand nahi',
        'khush nahi', 'satisfy nahi'
    ]
    
    for pattern in dissatisfaction_patterns:
        if pattern in message_lower:
            return True
    
    # Additional checks for common dissatisfaction structures
    if any(phrase in message_lower for phrase in ['not satisfied', 'not happy', 'not good']):
        if any(word in message_lower for word in ['with', 'about', 'regarding']):
            return True
    
    return False

def get_dissatisfaction_response(user_language: str = 'english') -> str:
    """Generate empathetic response for user dissatisfaction"""
    import random
    
    if user_language == 'hindi':
        responses = [
            "Maafi chahta hun ki main aapki help nahi kar saka. Main phir se try karunga Fasc Ai ke services ke bare mein.",
            "Main samajh gaya ki aap satisfied nahi hain. Main aapki better help kar sakta hun.",
            "Sorry for the confusion. Main aapki better help kar sakta hun Fasc Ai ke bare mein."
        ]
    else:
        responses = [
            "I apologize that my response wasn't helpful. Let me assist you better with Fasc Ai's services.",
            "I understand you're not satisfied. Let me help you more effectively with Fasc Ai's offerings.",
            "I'm sorry for the confusion. I can provide better assistance with Fasc Ai's services."
        ]
    
    return random.choice(responses)

def is_complaint(message: str) -> bool:
    """Check if user is complaining or expressing dissatisfaction"""
    message_lower = message.lower().strip()
    complaint_patterns = [
        'not happy', 'unhappy', 'disappointed', 'dissatisfied', 'not satisfied',
        'poor service', 'bad service', 'terrible service', 'worst service',
        'complaint', 'complain', 'issue with your', 'problem with your',
        'bad experience', 'poor support', 'terrible support', 'poor experience',
        'not satisfied', 'unsatisfied', 'let down', 'frustrated with your',
        'problem with fasc', 'issue with fasc', 'fasc ai is bad', 'fasc ai poor'
    ]
    
    for pattern in complaint_patterns:
        if pattern in message_lower:
            return True
    
    return False

# Client Identity Detection
def is_client_identity(message: str) -> bool:
    """Check if user is identifying themselves as a client"""
    message_lower = message.lower().strip()
    client_identity_patterns = [
        'i am your client', 'i\'m your client', 'i am a client', 'i\'m a client',
        'i am an existing client', 'i\'m an existing client', 
        'we are your client', 'we\'re your client', 'we are clients', 'we\'re clients',
        'i am your customer', 'i\'m your customer', 'i am a customer', 'i\'m a customer',
        'we are your customer', 'we\'re your customer',
        'i work with you', 'we work with you', 'we work with fasc',
        'i work with fasc', 'we use your service', 'i use your service',
        'i am already a client', 'i\'m already a client',
        'we are already a client', 'we\'re already a client',
        'existing client here', 'current client here',
        'i need support', 'i need help with my project',
        'we need support', 'we need help with our project'
    ]
    
    for pattern in client_identity_patterns:
        if pattern in message_lower:
            return True
    
    return False
def get_client_identity_response(user_language: str = 'english') -> str:
    """Generate welcoming response for existing clients"""
    import random
    
    if user_language == 'hindi':
        responses = [
            "Dhanyavaad Fasc Ai ke saath kaam karne ke liye! Main aapki madad kar sakta hun aapke project ya services ke saath.",
            "Humari valued client hone ke liye shukriya! Main aapke project ya account se related koi bhi sawal ka jawab de sakta hun.",
            "Aapka swagat hai! Main yahan hun aapki madad karne ke liye aapke project ya services ke bare mein."
        ]
    else:
        responses = [
            "Thank you for being a valued Fasc Ai client! I'm here to assist you with questions about your project, services, or account.",
            "Welcome! I'm here to support you with your Fasc Ai services.",
            "Thank you for choosing Fasc Ai! I'm here to help with questions or support for your projects or services.",
            "Great to connect with you! As a Fasc Ai client, I'm here to provide any assistance you need.",
            "Welcome back! I'm here to support you with your Fasc Ai services and projects."
        ]
    
    return random.choice(responses)

# Project Query Detection
def is_project_query(message: str) -> bool:
    """Check if user is asking about projects/portfolio"""
    message_lower = message.lower().strip()
    project_query_patterns = [
        'tell me about your projects', 'what projects have you done', 
        'show me your projects', 'your projects', 'your portfolio',
        'tell me about your previous projects', 'previous projects',
        'completed projects', 'past projects', 'project portfolio',
        'what have you built', 'what have you developed',
        'show me what you\'ve built', 'examples of your work',
        'your work samples', 'case studies', 'project examples',
        'what kind of projects', 'types of projects',
        'project list', 'list of projects', 'project names',
        'i am asking about your projects', 'asking about projects',
        'tell me something about your projects', 'about your projects'
    ]
    
    for pattern in project_query_patterns:
        if pattern in message_lower:
            return True
    
    return False

def get_project_query_response(user_language: str = 'english') -> str:
    """Generate response listing specific projects"""
    import random
    
    if user_language == 'hindi':
        responses = [
            "Humne Grand Trio Sports, Wonder Land Garden, Funzoop, Tysley, Dorundo, Dog Walking, aur Matson Surgicals jaise projects deliver kiye hain.",
            "Fasc Ai ne 250+ projects deliver kiye hain including Grand Trio Sports, Tysley, Dorundo, Wonder Land Garden, Funzoop, Dog Walking, aur Matson Surgicals."
        ]
    else:
        responses = [
            "We've delivered projects including Grand Trio Sports, Wonder Land Garden, Funzoop, Tysley, Dorundo, Dog Walking, and Matson Surgicals across various industries.",
            "Our portfolio includes Grand Trio Sports, Tysley, Dorundo, Wonder Land Garden, Funzoop, Dog Walking, and Matson Surgicals.",
            "We've built Grand Trio Sports, eCommerce platforms like Wonder Land Garden and Funzoop, Tysley AI chat platform, Dorundo scooter rental, Dog Walking services, and Matson Surgicals healthcare site.",
            "Among our 250+ projects are Grand Trio Sports, Wonder Land Garden, Funzoop, Tysley, Dorundo, Dog Walking, and Matson Surgicals."
        ]
    
    return random.choice(responses)

def get_complaint_response() -> str:
    """Generate empathetic response for complaints with support email"""
    import random
    responses = [
        "I apologize for your experience. Please share your concerns with our support team at support@fascai.com so we can address this immediately. For general inquiries, email info@fascai.com.",
        "I'm sorry to hear about your experience. We take all feedback seriously. Email your concerns to support@fascai.com and our team will prioritize resolving this.",
        "I apologize for any inconvenience. Please contact our support team at support@fascai.com to report this issue. For other inquiries, email info@fascai.com."
    ]
    return random.choice(responses)

# NEW: Support query detection
def is_support_query(message: str) -> bool:
    """Check if user is asking about support availability (24/7, response time, etc.)"""
    message_lower = message.lower().strip()
    support_patterns = [
        '24/7 support', '24x7 support', '24 7 support',
        'support hours', 'support time', 'support available',
        'when is support available', 'do you have support',
        'support schedule', 'support timing', 'support availability',
        'response time', 'support response', 'support response time',
        'when can i get support', 'is support available',
        'support contact', 'support team available'
    ]
    return any(pattern in message_lower for pattern in support_patterns)

# NEW: Specific project query detection
def is_specific_project_query(message: str) -> bool:
    """Check if user is asking about a specific project (e.g., 'dog walking project', 'funzoop project')"""
    message_lower = message.lower().strip()
    
    # Known project names
    project_names = [
        'dog walking', 'funzoop', 'dorundo', 'tysley',
        'grand trio sports', 'wonder land garden', 'matson surgicals'
    ]
    
    # Check if query mentions "project" + project name OR "tell me about" + project name
    has_project_keyword = any(word in message_lower for word in ['project', 'tell me about', 'about your', 'about the'])
    has_project_name = any(name in message_lower for name in project_names)
    
    # Also check for patterns like "dog walking project" or "tell me about funzoop"
    project_patterns = [
        f'{name} project' for name in project_names
    ] + [
        f'tell me about {name}' for name in project_names
    ] + [
        f'something about {name}' for name in project_names
    ]
    
    return (has_project_keyword and has_project_name) or any(pattern in message_lower for pattern in project_patterns)

# NEW: Company stats query detection
def is_company_stats_query(message: str) -> bool:
    """Check if user is asking about company statistics (project count, services count, etc.)"""
    message_lower = message.lower().strip()
    
    # Direct patterns for stats queries
    stats_patterns = [
        'how many projects', 'project count', 'number of projects',
        'how many services', 'service count', 'number of services',
        'how many clients', 'client count', 'number of clients',
        'how many employees', 'team size', 'number of employees',
        'projects completed', 'completed projects', 'projects you have',
        'projects you have completed', 'total projects', 'total services',
        'how many projects you have', 'how many services do you'
    ]
    
    # Count keywords + stats keywords combination
    count_keywords = ['how many', 'count', 'number of', 'total', 'how much']
    stats_keywords = ['projects', 'services', 'clients', 'employees', 'team']
    
    # Check for direct patterns
    if any(pattern in message_lower for pattern in stats_patterns):
        return True
    
    # Check for combination of count + stats keywords
    has_count_keyword = any(ck in message_lower for ck in count_keywords)
    has_stats_keyword = any(sk in message_lower for sk in stats_keywords)
    
    # Also check for "how many" + "provided" or "offer" (services count)
    if 'how many' in message_lower and any(word in message_lower for word in ['services', 'service', 'provided', 'offer', 'provide']):
        return True
    
    return has_count_keyword and has_stats_keyword

# NEW: Comparison query detection
def is_comparison_query(message: str) -> bool:
    """Check if user is asking about company comparison or competitive advantages"""
    message_lower = message.lower().strip()
    comparison_patterns = [
        'what makes you different', 'what makes your company different',
        'what makes you unique', 'what makes your company unique',
        'competitors', 'vs', 'versus', 'compare', 'comparison',
        'difference', 'differences', 'why choose you',
        'why you', 'why should i choose', 'advantage', 'advantages',
        'better than', 'why you are better', 'what sets you apart',
        'unique selling point', 'usp', 'competitive advantage',
        'how are you different', 'how do you differ'
    ]
    return any(pattern in message_lower for pattern in comparison_patterns)

# NEW: Industry query detection
def is_industry_query(message: str) -> bool:
    """Check if user is asking about industries served"""
    message_lower = message.lower().strip()
    industry_patterns = [
        'what industries', 'which industries', 'industries served',
        'industries do you serve', 'what sectors', 'which sectors',
        'industries you serve', 'sectors you serve',
        'what industries do you work with', 'which industries do you work with',
        'what industries are you in', 'industries you work in'
    ]
    return any(pattern in message_lower for pattern in industry_patterns)

# NEW: Response generation functions for new query types
async def generate_support_response(message: str, user_language: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate response for support availability queries using RAG"""
    # Search ChromaDB for support information
    search_results = search_chroma(message, COLLECTION_NAME, n_results=3)
    
    context = ""
    if search_results:
        context = "\n\n".join([result.get('content', '') for result in search_results])
    
    # Build context section
    if context and context.strip():
        context_section = f"""
        CRITICAL: The context provided below contains information about Fasc Ai's support services.
        You MUST base your answer PRIMARILY on this context.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about support availability (24/7 support, response time, etc.).
        Provide accurate information based on the context above. If context mentions 24/7 support, mention it.
        If no specific support hours are mentioned, state that support is available and mention contact info.
        """
    else:
        context_section = """
        Note: Use your general knowledge about Fasc Ai's support services.
        The user is asking about support availability. Provide information about support availability.
        """
    
    language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
    
    system_prompt = f"""
{context_section}

        You are an AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

        CRITICAL RULES:
        1. CONTEXT USAGE: When context is provided, use it to answer accurately.
        2. SUPPORT INFORMATION: Answer about support availability (24/7, response time, etc.).
        3. Keep response SHORT (1-2 sentences, 100 tokens max).
        4. NEVER ask follow-up questions.
        5. Be friendly and professional.
        6. LANGUAGE: {language_instruction}
        7. If context mentions 24/7 support, confirm it. If not, state that support is available.
        
        Example: "Yes, we offer 24/7 support for our IT solutions. You can reach us at info@fascai.com or +91-9958755444."
        """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {await _get_active_groq_key()}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                reply = result['choices'][0]['message']['content'].strip()
                return strip_markdown(reply)
    except Exception as e:
        logger.error(f"Error generating support response: {e}")
    
    # Fallback response
    return "Yes, we offer comprehensive support for our IT solutions. You can reach us at info@fascai.com or +91-9958755444."

async def generate_specific_project_response(message: str, user_language: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate response for specific project queries using RAG"""
    # Search ChromaDB for project information
    search_results = search_chroma(message, COLLECTION_NAME, n_results=3)
    
    context = ""
    if search_results:
        context = "\n\n".join([result.get('content', '') for result in search_results])
    
    # Build context section
    if context and context.strip():
        context_section = f"""
        CRITICAL: The context provided below contains information about Fasc Ai's completed projects.
        You MUST base your answer PRIMARILY on this context.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about a specific project (Dog Walking, Funzoop, Dorundo, etc.).
        Provide detailed information about the project based on the context above.
        Mention what was built, technologies used, and outcomes if available in context.
        """
    else:
        context_section = """
        Note: Use your general knowledge about Fasc Ai's projects.
        The user is asking about a specific project. Provide information about the project.
        """
    
    language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
    
    system_prompt = f"""
{context_section}

        You are an AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

        CRITICAL RULES:
        1. CONTEXT USAGE: When context is provided, use it to answer accurately.
        2. PROJECT INFORMATION: Provide details about the specific project mentioned.
        3. Keep response SHORT (1-2 sentences, 100 tokens max).
        4. NEVER ask follow-up questions.
        5. Be friendly and professional.
        6. LANGUAGE: {language_instruction}
        7. Mention project name, what was built, and key features if available.
        
        Example: "We successfully implemented the Dog Walking pet services platform, providing seamless user experience and efficient operations management."
        """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {await _get_active_groq_key()}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                reply = result['choices'][0]['message']['content'].strip()
                return strip_markdown(reply)
    except Exception as e:
        logger.error(f"Error generating specific project response: {e}")
    
    # Fallback - extract project name and give generic response
    message_lower = message.lower()
    project_map = {
        'dog walking': 'Dog Walking pet services platform',
        'funzoop': 'Funzoop e-commerce platform',
        'dorundo': 'Dorundo electric scooter rental platform',
        'tysley': 'Tysley AI chat platform',
        'grand trio sports': 'Grand Trio Sports cricket services',
        'wonder land garden': 'Wonder Land Garden e-commerce site',
        'matson surgicals': 'Matson Surgicals healthcare website'
    }
    
    for key, value in project_map.items():
        if key in message_lower:
            return f"We successfully implemented the {value}, providing seamless user experience and efficient operations management."
    
    return "We've successfully delivered this project as part of our 250+ completed projects across various industries."

async def generate_company_stats_response(message: str, user_language: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate response for company statistics queries using RAG"""
    # Search ChromaDB with multiple query variations for better matching
    queries = [
        message,
        f"Fasc Ai {message}",
        f"Fasc Ai projects completed count statistics",
        f"Fasc Ai portfolio statistics",
        f"Fasc Ai 250+ projects completed"
    ]
    
    # Search with all variations and combine results
    all_results = []
    for query in queries[:3]:  # Use first 3 variations to avoid too many searches
        results = search_chroma(query, COLLECTION_NAME, n_results=2)
        if results:
            all_results.extend(results)
    
    # Remove duplicates based on content
    seen_content = set()
    unique_results = []
    for result in all_results:
        content = result.get('content', '')
        if content and content not in seen_content:
            seen_content.add(content)
            unique_results.append(result)
    
    context = ""
    if unique_results:
        context = "\n\n".join([result.get('content', '') for result in unique_results[:3]])
    
    # Build context section
    if context and context.strip():
        context_section = f"""
        CRITICAL: The context provided below contains information about Fasc Ai's company statistics.
        You MUST base your answer PRIMARILY on this context.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about company statistics (project count, services count, etc.).
        Provide accurate information based on the context above. If context mentions "250+ projects", mention it.
        If asking about services, list the services we provide.
        """
    else:
        context_section = """
        Note: Use your general knowledge about Fasc Ai's company statistics.
        The user is asking about company statistics. Provide accurate information.
        """
    
    language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
    
    # Determine what stats are being asked
    message_lower = message.lower()
    if 'project' in message_lower:
        stats_type = "projects completed"
        default_info = "We've completed 250+ projects including Grand Trio Sports, Funzoop, Wonder Land Garden, Tysley, Dorundo, Dog Walking, and Matson Surgicals."
    elif 'service' in message_lower:
        stats_type = "services provided"
        default_info = "We provide cloud computing, ERP, CRM, AI solutions, and IoT services."
    elif 'client' in message_lower:
        stats_type = "clients"
        default_info = "Our clients include MOF, Max Life, Lenovo, Medanta, Videocon, and Saarte."
    else:
        stats_type = "company statistics"
        default_info = "We've completed 250+ projects across various industries with 19+ years of experience."
    
    system_prompt = f"""
{context_section}

        You are an AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

        CRITICAL RULES:
        1. CONTEXT USAGE: When context is provided, use it to answer accurately.
        2. STATISTICS INFORMATION: Answer about {stats_type} based on the context.
        3. Keep response SHORT (1-2 sentences, 100 tokens max).
        4. NEVER ask follow-up questions.
        5. Be friendly and professional.
        6. LANGUAGE: {language_instruction}
        7. If context has specific numbers (like "250+ projects"), use them. If not, use general information.
        
        Default information if context doesn't have specifics: {default_info}
        """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {await _get_active_groq_key()}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                reply = result['choices'][0]['message']['content'].strip()
                return strip_markdown(reply)
    except Exception as e:
        logger.error(f"Error generating company stats response: {e}")
    
    # Fallback response
    return default_info

async def generate_comparison_response(message: str, user_language: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate response for comparison queries using RAG"""
    # Search ChromaDB for company advantages
    search_results = search_chroma(message, COLLECTION_NAME, n_results=3)
    
    context = ""
    if search_results:
        context = "\n\n".join([result.get('content', '') for result in search_results])
    
    # Build context section
    if context and context.strip():
        context_section = f"""
        CRITICAL: The context provided below contains information about Fasc Ai's competitive advantages.
        You MUST base your answer PRIMARILY on this context.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking what makes Fasc Ai different from competitors.
        Provide information about company advantages, experience, expertise, and unique selling points based on context.
        """
    else:
        context_section = """
        Note: Use your general knowledge about Fasc Ai's competitive advantages.
        The user is asking what makes the company different. Provide information about advantages.
        """
    
    language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
    
    system_prompt = f"""
{context_section}

        You are an AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

        CRITICAL RULES:
        1. CONTEXT USAGE: When context is provided, use it to answer accurately.
        2. COMPARISON INFORMATION: Answer about what makes Fasc Ai different/unique.
        3. Keep response SHORT (1-2 sentences, 100 tokens max).
        4. NEVER ask follow-up questions.
        5. Be friendly and professional.
        6. LANGUAGE: {language_instruction}
        7. Mention key advantages: 19+ years experience, 250+ projects, expertise in IT solutions, client satisfaction.
        
        Example: "Our 19+ years of experience, 250+ successfully delivered projects, and focus on client satisfaction set us apart. We specialize in cloud computing, ERP, CRM, AI solutions, and IoT implementations."
        """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {await _get_active_groq_key()}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                reply = result['choices'][0]['message']['content'].strip()
                return strip_markdown(reply)
    except Exception as e:
        logger.error(f"Error generating comparison response: {e}")
    
    # Fallback response
    return "Our 19+ years of experience, 250+ successfully delivered projects, and focus on client satisfaction set us apart. We specialize in cloud computing, ERP, CRM, AI solutions, and IoT implementations."

async def generate_industry_response(message: str, user_language: str, conversation_history: List[Dict[str, str]]) -> str:
    """Generate response for industry queries using RAG"""
    # Search ChromaDB for industry information
    search_results = search_chroma(message, COLLECTION_NAME, n_results=3)
    
    context = ""
    if search_results:
        context = "\n\n".join([result.get('content', '') for result in search_results])
    
    # Build context section
    if context and context.strip():
        context_section = f"""
        CRITICAL: The context provided below contains information about industries served by Fasc Ai.
        You MUST base your answer PRIMARILY on this context.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about industries served.
        Provide information about industries/sectors served based on the context above.
        """
    else:
        context_section = """
        Note: Use your general knowledge about industries served by Fasc Ai.
        The user is asking about industries served. Provide information about industries.
        """
    
    language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
    
    system_prompt = f"""
{context_section}

        You are an AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

        CRITICAL RULES:
        1. CONTEXT USAGE: When context is provided, use it to answer accurately.
        2. INDUSTRY INFORMATION: Answer about industries served.
        3. Keep response SHORT (1-2 sentences, 100 tokens max).
        4. NEVER ask follow-up questions.
        5. Be friendly and professional.
        6. LANGUAGE: {language_instruction}
        7. Mention industries like finance, insurance, healthcare, retail, manufacturing, technology if available in context.
        
        Example: "We serve various industries including finance, insurance, healthcare, retail, manufacturing, and technology sectors."
        """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": message})
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {await _get_active_groq_key()}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.7
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                reply = result['choices'][0]['message']['content'].strip()
                return strip_markdown(reply)
    except Exception as e:
        logger.error(f"Error generating industry response: {e}")
    
    # Fallback response
    return "We serve various industries including finance, insurance, healthcare, retail, manufacturing, and technology sectors."

# Conversation Context Management
def is_follow_up_response(message: str, last_bot_response: str = "") -> bool:
    """Check if message is a follow-up response to a previous bot question - DISABLED to prevent unwanted follow-ups"""
    # DISABLED: Follow-up detection disabled to prevent unwanted follow-up questions
    # All responses will be treated as regular queries for better user experience
    return False
def is_definition_query(message: str) -> bool:
    """Check if message is asking for a definition (what is, what are, explain)"""
    import re
    message_lower = message.lower().strip()
    
    # Definition query patterns
    definition_patterns = [
        r'^what is\s+',
        r'^what are\s+',
        r'^what\'s\s+',
        r'^explain\s+',
        r'^tell me what\s+',
        r'^define\s+',
        r'^what do you mean by\s+',
        r'^what does\s+.*\s+mean',
        r'^can you explain\s+',
        r'^can you tell me what\s+'
    ]
    
    for pattern in definition_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False

def verify_service_provided(user_query: str, context: str, search_results: List[Dict[str, Any]] = None) -> bool:
    """
    Semantic verification: Check if context actually contains EXACT service mentioned in query
    Returns True only if service is explicitly mentioned in context, False otherwise
    Now supports synonym matching (e.g., "AI services" = "AI implementations" = "AI solutions")
    """
    import re
    
    if not context or not user_query:
        return False
    
    # Service synonyms dictionary - maps service categories to their synonyms
    SERVICE_SYNONYMS = {
        'ai': [
            'ai', 'artificial intelligence', 'ai implementation', 'ai implementations',
            'ai solution', 'ai solutions', 'ai service', 'ai services', 'ai platform',
            'ai platforms', 'ai integration', 'ai integrations', 'ai deployment', 'ai deployments',
            'ai consulting', 'ai strategy', 'machine learning', 'ml', 'ml model', 'ml models',
            'machine learning model', 'machine learning models', 'model training', 'model fine-tuning',
            'model tuning', 'deep learning', 'neural network', 'neural networks', 'deep neural network',
            'deep neural networks', 'representation learning', 'hierarchical learning', 'deep structured learning',
            'deep machine learning', 'computer vision', 'machine vision', 'visual inspection',
            'automated visual inspection', 'image processing', 'image analysis', 'object detection',
            'pattern recognition', 'natural language processing', 'nlp', 'conversational ai',
            'virtual agent', 'virtual agents', 'chatbot', 'chatbots', 'virtual assistant', 'virtual assistants',
            'language understanding', 'intelligent automation', 'ai automation', 'automation', 'hyperautomation',
            'smart automation', 'cognitive automation', 'supervised learning', 'unsupervised learning',
            'reinforcement learning', 'self-supervised learning', 'transfer learning', 'generative ai',
            'foundation model', 'foundation models', 'large language model', 'large language models',
            'llm', 'llms', 'generative model', 'generative models', 'diffusion model', 'diffusion models',
            'ml ops', 'mlops', 'model optimization', 'model monitoring', 'predictive analytics',
            'decision intelligence', 'recommendation engine', 'recommendation engines', 'forecasting model',
            'forecasting models', 'edge ai', 'robotic process automation', 'rpa', 'robotics automation',
            'autonomous systems'
        ],
        'cloud': ['cloud', 'cloud computing', 'cloud hosting', 'cloud service', 'cloud services', 'cloud solution', 'cloud solutions', 'hosting', 'cloud infrastructure'],
        'erp': ['erp', 'enterprise resource planning', 'erp system', 'erp systems', 'erp implementation', 'erp implementations', 'erp service', 'erp services'],
        'crm': ['crm', 'customer relationship management', 'crm system', 'crm systems', 'crm platform', 'crm platforms', 'crm service', 'crm services'],
        'iot': ['iot', 'internet of things', 'iot solution', 'iot solutions', 'iot implementation', 'iot implementations', 'iot service', 'iot services'],
        'website': ['website', 'web development', 'web design', 'website development', 'website services', 'web services', 'web development services', 'website design', 'web application', 'web applications', 'ecommerce', 'e-commerce', 'ecommerce development', 'e-commerce development']
    }
    
    # Extract service name from user query (e.g., "do you provide X services" -> "X")
    query_lower = user_query.lower()
    context_lower = context.lower()
    
    # Common non-IT services that should be automatically declined
    non_it_services = [
        'hardware installation', 'hardware maintenance', 'hardware repair',
        'courier service', 'courier services', 'delivery service', 'logistics service',
        'beauty service', 'beauty services', 'cosmetic service',
        'food service', 'food services', 'catering service',
        'lifestyle service', 'lifestyle services',
        'home maintenance', 'home repair', 'home improvement',
        'financial service', 'financial services', 'banking service', 'insurance service',
        'health and fitness', 'fitness service', 'gym service',
        'family service', 'family services', 'childcare service'
    ]
    
    # Check if query mentions non-IT services - automatically decline
    for non_it_service in non_it_services:
        if non_it_service in query_lower:
            # Extra check: verify if context actually contains this specific service
            service_keywords = non_it_service.split()
            # Check if ALL keywords appear together in context (exact match)
            if all(keyword in context_lower for keyword in service_keywords):
                # Check if context says we provide this service explicitly
                provide_patterns = [
                    f'provide {non_it_service}',
                    f'offer {non_it_service}',
                    f'{non_it_service} we',
                    f'our {non_it_service}',
                    f'{non_it_service} service'
                ]
                if any(pattern in context_lower for pattern in provide_patterns):
                    # If context explicitly says we provide it, allow it
                    continue
            # If not explicitly mentioned, decline
            logger.info(f"Non-IT service detected and declined: {non_it_service}")
            return False
    
    # Extract potential service names from query
    # Pattern: "do you provide X" or "do you offer X" or "X services"
    service_patterns = [
        r'provide\s+([^?]+?)(?:\s+service|services)?',
        r'offer\s+([^?]+?)(?:\s+service|services)?',
        r'([^?]+?)\s+service',
        r'([^?]+?)\s+services'
    ]
    
    extracted_services = []
    seen_candidates = set()
    service_candidates: List[Tuple[str, Optional[str]]] = []
    for pattern in service_patterns:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            service_name = match.strip()
            if service_name and len(service_name) > 3:  # Valid service name
                # Normalize: Remove "services" suffix if it exists (for better synonym matching)
                # Only remove if "services" is at the end, not in the middle
                normalized_service = service_name.strip()
                if normalized_service.endswith(' services'):
                    normalized_service = normalized_service[:-9].strip()  # Remove " services"
                elif normalized_service.endswith(' service'):
                    normalized_service = normalized_service[:-8].strip()  # Remove " service"
                
                if normalized_service and normalized_service not in seen_candidates:
                    service_candidates.append((normalized_service, None))
                    seen_candidates.add(normalized_service)
    
    if not service_candidates:
        # Priority check: If query contains "web", check 'website' category first for faster processing
        if 'web' in query_lower and 'website' in SERVICE_SYNONYMS:
            for synonym in SERVICE_SYNONYMS['website']:
                if synonym in query_lower and synonym not in seen_candidates:
                    service_candidates.append((synonym, 'website'))
                    seen_candidates.add(synonym)
                    break
        
        # Check other categories (including website if not already checked)
        for category, synonyms in SERVICE_SYNONYMS.items():
            # Skip website if already checked above
            if category == 'website' and any(cat == 'website' for _, cat in service_candidates):
                continue
            for synonym in synonyms:
                if synonym in query_lower and synonym not in seen_candidates:
                    service_candidates.append((synonym, category))
                    seen_candidates.add(synonym)
                    break
        if not service_candidates:
            return False
    
    # Check if context contains service name (with synonym support)
    for service_name, predefined_category in service_candidates:
        # Remove common words
        service_words = [w for w in service_name.split() if w not in ['the', 'a', 'an', 'and', 'or', 'do', 'you', 'provide', 'offer']]
        if not service_words:
            continue
        
        # Find which service category this belongs to (if any)
        service_category = predefined_category
        service_name_lower = service_name.lower().strip()
        if service_category is None:
            # Improved category detection: Check for better partial phrase matching
            for category, synonyms in SERVICE_SYNONYMS.items():
                # Check if any synonym matches the extracted service name
                for synonym in synonyms:
                    synonym_lower = synonym.lower()
                    # Check for exact match or substring match (bidirectional)
                    if (synonym_lower == service_name_lower or 
                        synonym_lower in service_name_lower or 
                        service_name_lower in synonym_lower):
                        service_category = category
                        break
                    # Additional check: If service name contains key words from synonym
                    # This helps with "web development" matching "website development"
                    synonym_words = set(synonym_lower.split())
                    service_words = set(service_name_lower.split())
                    # If all key words from synonym are in service name (for multi-word synonyms)
                    if len(synonym_words) > 1 and synonym_words.issubset(service_words):
                        service_category = category
                        break
                if service_category:
                    break
        
        # Get all synonyms for this service category (if found)
        synonyms_to_check = []
        if service_category:
            synonyms_to_check = SERVICE_SYNONYMS[service_category]
        else:
            # If no category found, use the original service name
            synonyms_to_check = [service_name]
        
        # Check if context contains any of the synonyms with service indicators
        for synonym in synonyms_to_check:
            synonym_lower = synonym.lower()
            
            # Flexible matching: First check if synonym exists directly in context (faster and more flexible)
            # This helps when context has "web development" but not exact "provide web development"
            if synonym_lower in context_lower:
                # Additional validation: Check if it's not part of a negative phrase
                negative_phrases = ['not provide', 'do not', "don't", 'cannot', 'unable to', 'no ']
                context_words_around = context_lower[max(0, context_lower.find(synonym_lower) - 50):context_lower.find(synonym_lower) + len(synonym_lower) + 50]
                if not any(neg in context_words_around for neg in negative_phrases):
                    logger.info(f"Service verified in context via flexible matching: {service_name} (matched synonym: {synonym})")
                    return True
            
            # Word-set matching: Check if key words from service name match ANY synonym's words in context
            # This helps when "web development" needs to match "website development" in context
            # Example: "web development" words ["web", "development"] should match "website development"
            # Strategy: Check if service name's key words are present in context, AND if any synonym's key words are also present
            if len(service_name_lower.split()) > 1:
                service_name_words = set(service_name_lower.split())
                # Remove common stop words that don't add meaning
                stop_words = {'the', 'a', 'an', 'and', 'or', 'of', 'for', 'in', 'on', 'at', 'to', 'is', 'are', 'was', 'were', 'do', 'you', 'provide', 'offer', 'services', 'service'}
                service_key_words = {w for w in service_name_words if w not in stop_words and len(w) > 2}
                
                if service_key_words:
                    # Check if service name's key words are present in context (substring match for flexibility)
                    # "web" will match in "website", "development" will match in "website development"
                    service_words_found = sum(1 for word in service_key_words if word in context_lower)
                    
                    # Also check if synonym's key words are in context (for cross-synonym matching)
                    synonym_key_words = set()
                    if len(synonym_lower.split()) > 1:
                        synonym_words = set(synonym_lower.split())
                        synonym_key_words = {w for w in synonym_words if w not in stop_words and len(w) > 2}
                    
                    # Combine both sets for comprehensive matching
                    all_key_words = service_key_words | synonym_key_words
                    if all_key_words:
                        all_words_found = sum(1 for word in all_key_words if word in context_lower)
                        # If at least 60% of key words are found, consider it a match (lowered for better matching)
                        # This allows "web development" to match "website development" even if "web" is substring of "website"
                        if all_words_found >= len(all_key_words) * 0.6:
                            # Additional validation: Check if it's not part of a negative phrase
                            # Check around where key words appear in context
                            key_word_positions = [context_lower.find(word) for word in all_key_words if word in context_lower]
                            if key_word_positions:
                                min_pos = min(key_word_positions)
                                max_pos = max(key_word_positions) + max(len(w) for w in all_key_words)
                                context_words_around = context_lower[max(0, min_pos - 50):min(len(context_lower), max_pos + 50)]
                                negative_phrases = ['not provide', 'do not', "don't", 'cannot', 'unable to', 'no ']
                                if not any(neg in context_words_around for neg in negative_phrases):
                                    logger.info(f"Service verified in context via word-set matching: {service_name} (matched synonym: {synonym}, key words: {all_key_words}, found: {all_words_found}/{len(all_key_words)})")
                                    return True
            
            # Service indicators to check in context (exact phrases - more strict)
            service_indicators = [
                f'provide {synonym_lower}',
                f'offer {synonym_lower}',
                f'{synonym_lower} service',
                f'{synonym_lower} services',
                f'our {synonym_lower}',
                f'{synonym_lower} we',
                f'{synonym_lower} solutions',
                f'{synonym_lower} implementations',
                f'{synonym_lower} implementation',
                f'{synonym_lower} platform',
                f'{synonym_lower} platforms',
                f'{synonym_lower} capability',
                f'{synonym_lower} capabilities',
                f'{synonym_lower} expertise',
                f'{synonym_lower} offering',
                f'{synonym_lower} offerings',
                f'{synonym_lower} practice',
                f'{synonym_lower} program',
                f'{synonym_lower} programs',
                f'{synonym_lower} team',
                f'{synonym_lower} teams'
            ]
            
            # Check if any indicator suggests we provide this service
            if any(indicator in context_lower for indicator in service_indicators):
                logger.info(f"Service verified in context: {service_name} (matched synonym: {synonym})")
                return True
            
            synonym_pattern = re.escape(synonym_lower)
            suffix_pattern = r'(solutions?|services?|implementations?|platforms?|capabilities?|offerings?|expertise|practice|programs?|teams?)'
            if re.search(rf'{synonym_pattern}\s+{suffix_pattern}', context_lower):
                logger.info(f"Service verified in context via suffix match: {service_name} (matched synonym: {synonym})")
                return True
            if re.search(rf'{suffix_pattern}\s+for\s+{synonym_pattern}', context_lower):
                logger.info(f"Service verified in context via prefix match: {service_name} (matched synonym: {synonym})")
                return True
        
        # Fallback: Check if ALL key words from service name appear in context (original logic)
        key_words_found = sum(1 for word in service_words if word in context_lower)
        
        # For strict matching: at least 70% of key words should be present
        if key_words_found >= len(service_words) * 0.7:
            # Additional check: context should contain service-related keywords
            service_indicators = [
                f'provide {service_name}',
                f'offer {service_name}',
                f'{service_name} service',
                f'{service_name} services',
                f'our {service_name}',
                f'{service_name} we'
            ]
            
            # Check if any indicator suggests we provide this service
            if any(indicator in context_lower for indicator in service_indicators):
                logger.info(f"Service verified in context: {service_name}")
                return True
    
    # If no match found, decline
    logger.info(f"Service not explicitly verified in context for query: {user_query}")
    return False
def extract_service_from_response(last_bot_response: str) -> tuple:
    """
    Multi-layer service extraction from bot response
    Returns: (service_name, confidence_score)
    """
    import re
    last_response_lower = last_bot_response.lower()
    
    # Layer 1: Direct pattern extraction (highest confidence)
    direct_patterns = {
        'crm': [r'\bcrm\s+(?:implementation|services?|solutions?|systems?)', r'\bcustomer\s+relationship\s+management'],
        'erp': [r'\berp\s+(?:implementation|services?|solutions?|systems?)', r'\benterprise\s+resource\s+planning'],
        'ai': [r'\bai\s+(?:solutions?|services?|implementation)', r'\bartificial\s+intelligence'],
        'cloud': [r'\bcloud\s+(?:computing|services?|solutions?|hosting)', r'\bcloud\s+transformation'],
        'iot': [r'\biot\s+(?:solutions?|services?|implementation|devices?)', r'\binternet\s+of\s+things'],
        'aws': [r'\baws\s+(?:services?|solutions?|cloud)', r'\bamazon\s+web\s+services'],
        'website': [r'\bwebsite\s+(?:development|services?)', r'\bweb\s+development'],
        'ecommerce': [r'\be-?commerce\s+(?:solutions?|services?)', r'\becommerce\s+(?:platforms?|solutions?)'],
    }
    
    for service, patterns in direct_patterns.items():
        for pattern in patterns:
            if re.search(pattern, last_response_lower):
                logger.info(f"Layer 1: Extracted service '{service}' from direct pattern")
                return (service, 0.95)
    
    # Layer 2: Keyword-based extraction (medium confidence)
    service_keywords = {
        'crm': ['crm', 'customer relationship'],
        'erp': ['erp', 'enterprise resource planning'],
        'ai': ['ai', 'artificial intelligence', 'machine learning'],
        'cloud': ['cloud computing', 'cloud hosting', 'cloud services', 'aws', 'azure'],
        'iot': ['iot', 'internet of things'],
        'website': ['website', 'web development', 'web design'],
        'ecommerce': ['ecommerce', 'e-commerce', 'shopify'],
    }
    
    for service, keywords in service_keywords.items():
        if any(keyword in last_response_lower for keyword in keywords):
            # Check if it's part of a question about that service
            if any(phrase in last_response_lower for phrase in [
                f'{service}', f'{keywords[0]}', 'about', 'regarding', 'regarding our'
            ]):
                logger.info(f"Layer 2: Extracted service '{service}' from keywords")
                return (service, 0.85)
    
    # Layer 3: HF Intent Detection (if available)
    if TRANSFORMERS_AVAILABLE:
        try:
            # Create smart query combining bot response + user intent
            smart_query = f"{last_bot_response}. User wants more information about: {last_response_lower}"
            hf_result = detect_intent_with_hf(smart_query)
            
            if hf_result and hf_result.get('confidence', 0) >= 0.6:
                detected_intent = hf_result.get('intent', '')
                confidence = hf_result.get('confidence', 0)
                
                # Map HF intents to services
                intent_to_service = {
                    'crm_inquiry': ('crm', 0.80),
                    'erp_inquiry': ('erp', 0.80),
                    'cloud_inquiry': ('cloud', 0.80),
                    'iot_inquiry': ('iot', 0.80),
                    'ai_inquiry': ('ai', 0.80),
                    'service_inquiry': ('general', 0.70),
                }
                
                if detected_intent in intent_to_service:
                    service, base_confidence = intent_to_service[detected_intent]
                    final_confidence = min(0.85, base_confidence * confidence)
                    logger.info(f"Layer 3: Extracted service '{service}' from HF intent '{detected_intent}' with confidence {final_confidence:.2f}")
                    return (service, final_confidence)
        except Exception as e:
            logger.warning(f"HF intent detection failed in service extraction: {e}")
    
    # No service detected
    return (None, 0.0)

def get_follow_up_context(last_bot_response: str, user_response: str) -> Dict[str, Any]:
    """Determine the context for follow-up responses with multi-layer service extraction"""
    last_response_lower = last_bot_response.lower()
    
    # Extract service using multi-layer approach
    detected_service, service_confidence = extract_service_from_response(last_bot_response)
    
    # AI Benefits context (specific check first)
    if any(phrase in last_response_lower for phrase in ['ai can benefit', 'ai solutions', 'would you like to know more', 'how ai can benefit']):
        return {
            "type": "ai_benefits_followup",
            "original_question": "ai_benefits",
            "user_response": user_response.lower(),
            "last_bot_response": last_bot_response,
            "detected_service": "ai",
            "service_confidence": 0.9
        }
    
    # Service-specific follow-up (if service detected)
    if detected_service and service_confidence >= 0.7:
        return {
            "type": "service_followup",
            "original_question": f"{detected_service}_details",
            "user_response": user_response.lower(),
            "last_bot_response": last_bot_response,
            "detected_service": detected_service,
            "service_confidence": service_confidence
        }
    
    # Service information context (generic)
    if any(phrase in last_response_lower for phrase in ['our services', 'what services', 'tell you about our services']):
        return {
            "type": "service_info_followup", 
            "original_question": "service_info",
            "user_response": user_response.lower(),
            "last_bot_response": last_bot_response,
            "detected_service": detected_service if detected_service else None,
            "service_confidence": service_confidence
        }
    
    # Pricing context
    if any(phrase in last_response_lower for phrase in ['pricing', 'cost', 'price', 'packages']):
        return {
            "type": "pricing_followup",
            "original_question": "pricing",
            "user_response": user_response.lower(),
            "last_bot_response": last_bot_response,
            "detected_service": None,
            "service_confidence": 0.0
        }
    
    # Project context
    if any(phrase in last_response_lower for phrase in ['project', 'work with you', 'collaborate']):
        return {
            "type": "project_followup",
            "original_question": "project",
            "user_response": user_response.lower(),
            "last_bot_response": last_bot_response,
            "detected_service": None,
            "service_confidence": 0.0
        }
    
    # Default context with service info if available
    return {
        "type": "general_followup",
        "original_question": "unknown",
        "user_response": user_response.lower(),
        "last_bot_response": last_bot_response,
        "detected_service": detected_service if detected_service else None,
        "service_confidence": service_confidence
    }

def strip_markdown(text: str) -> str:
    """Strip markdown formatting from text"""
    import re
    
    # Remove bold formatting (**text** -> text)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Remove italic formatting (*text* -> text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove code formatting (`text` -> text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Remove link formatting ([text](url) -> text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Remove header formatting (# text -> text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    return text.strip()
def generate_rag_query_for_followup(followup_type: str, last_bot_response: str = "", detected_service: str = None) -> List[str]:
    """
    Generate smart RAG queries for different follow-up types
    Returns list of query variations to try
    """
    queries = []
    last_response_lower = last_bot_response.lower() if last_bot_response else ""
    
    if followup_type == "pricing_followup":
        queries = [
            "pricing packages costs pricing information quotes payment plans",
            "pricing models packages costs how much",
            "pricing information quote estimate",
            "cost pricing packages"
        ]
    
    elif followup_type == "project_followup":
        queries = [
            "project onboarding process collaboration steps getting started",
            "how to start project collaboration process",
            "project process requirements steps",
            "working together project collaboration"
        ]
    
    elif followup_type == "ai_benefits_followup":
        queries = [
            "AI benefits business transformation automation artificial intelligence advantages",
            "how AI can benefit business automation",
            "AI advantages benefits business",
            "artificial intelligence business benefits"
        ]
    
    elif followup_type == "service_info_followup":
        queries = [
            "IT services cloud ERP CRM AI IoT solutions offerings",
            "services offered solutions available",
            "what services do you offer",
            "services solutions offerings"
        ]
    
    elif followup_type == "service_followup" and detected_service:
        queries = [
            f"{detected_service} implementation services features benefits",
            f"{detected_service} solutions services",
            f"{detected_service} how it works benefits"
        ]
    
    elif followup_type == "general_followup":
        # Extract keywords from last bot response
        import re
        # Extract meaningful words (exclude common words)
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'about', 'with', 'for', 'from', 'to', 'in', 'on', 'at', 'by', 'of', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'how', 'what', 'which', 'who', 'why', 'you', 'your', 'yours', 'we', 'our', 'ours', 'they', 'their', 'them', 'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'mine', 'he', 'she', 'his', 'her', 'hers'}
        
        words = re.findall(r'\b\w+\b', last_response_lower)
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        if meaningful_words:
            # Use top 3-5 meaningful words
            keywords = ' '.join(meaningful_words[:5])
            queries = [keywords, last_bot_response[:100]]  # Try keywords and first 100 chars
        else:
            queries = [last_bot_response[:100]]
    
    return queries if queries else [last_bot_response[:100]]

async def generate_rag_based_response(
    followup_type: str,
    queries: List[str],
    last_bot_response: str,
    user_response: str,
    conversation_history: List[Dict[str, str]],
    user_language: str,
    detected_service: str = None
) -> str:
    """
    Universal function to generate RAG-based detailed responses for any follow-up type
    """
    try:
        # Try multiple query variations
        rag_results = []
        for query in queries[:3]:  # Try max 3 query variations
            results = search_chroma(query, COLLECTION_NAME, n_results=3)
            if results:
                rag_results.extend(results)
                if len(rag_results) >= 5:  # Collect enough results
                    break
        
        # Remove duplicates based on content
        seen_content = set()
        unique_results = []
        for result in rag_results:
            content_hash = hash(result.get('content', '')[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        rag_results = unique_results[:5]  # Limit to top 5 unique results
        
        # Build context from RAG results
        context_text = ""
        if rag_results:
            context_text = "\n\n".join([result.get('content', '') for result in rag_results])
            logger.info(f"Found {len(rag_results)} relevant documents for {followup_type}")
        
        # Generate response using Groq API
        language_instruction = "You MUST respond in Hindi only." if user_language == 'hindi' else "You MUST respond in English only."
        
        # Type-specific prompt templates - SHORT responses only (1-2 sentences)
        prompt_templates = {
            'pricing_followup': {
                'instruction': "The user wants pricing information. Provide brief, concise details about pricing models and how to get a quote (1-2 sentences only).",
                'include_contact': True
            },
            'project_followup': {
                'instruction': "The user wants to start a project. Provide brief information about the project process and how to get started (1-2 sentences only).",
                'include_contact': True
            },
            'ai_benefits_followup': {
                'instruction': "The user wants to know about AI benefits. Provide brief, key points about how AI can benefit businesses (1-2 sentences only).",
                'include_contact': False
            },
            'service_info_followup': {
                'instruction': "The user wants information about services. Provide brief overview of our IT solutions (1-2 sentences only).",
                'include_contact': False
            },
            'service_followup': {
                'instruction': "The user wants information about services. Provide brief, key details about the service (1-2 sentences only).",
                'include_contact': False
            },
            'general_followup': {
                'instruction': "The user wants more information. Provide brief, helpful information based on the context (1-2 sentences only).",
                'include_contact': False
            }
        }
        
        template = prompt_templates.get(followup_type, prompt_templates['general_followup'])
        
        # Customize instruction for service_followup with detected service
        if followup_type == 'service_followup' and detected_service:
            template = template.copy()  # Create a copy to avoid modifying the original
            template['instruction'] = f"The user wants detailed information about {detected_service.upper()} services. Provide comprehensive details about {detected_service.upper()} implementation process, features, capabilities, business benefits, use cases, and how it can help their business."
        
        context_section = ""
        if context_text:
            context_section = f"""
        CRITICAL: The context below contains specific information from Fasc Ai's knowledge base.
        You MUST base your answer PRIMARILY on this context. Provide detailed, comprehensive information.
        
        CONTEXT FROM KNOWLEDGE BASE:
        {context_text[:1500]}
        
        INSTRUCTIONS: {template['instruction']}
        """
        else:
            context_section = f"""
        Note: Provide comprehensive information based on your knowledge about Fasc Ai's services.
        {template['instruction']}
        """
        
        contact_info = ""
        if template.get('include_contact', False):
            contact_info = " For detailed information tailored to your specific needs, you can also contact info@fascai.com or visit fascai.com/contact."
        
        system_prompt = f"""
{context_section}
        
        You are an AI assistant for Fasc Ai Ventures Private Limited IT solutions company.
        
        CRITICAL RULES:
        1. The user just said "yes" or similar after you asked if they want to know more.
        2. Provide SHORT and COMPLETE information based on the context and instructions above.
        3. Be enthusiastic, helpful, and professional.
        4. CRITICAL: Keep it SHORT and COMPLETE - MUST be 1-2 sentences MAX with key points only. Maximum 100 tokens. DO NOT exceed this limit. Ensure your response is COMPLETE - end with a full sentence and period. Do NOT cut off mid-sentence.
        5. CRITICAL: NEVER ask follow-up questions. NEVER end responses with questions like "What would you like to know?", "What do you need help with?", "Would you like to know more?", etc. Just provide the information directly and end with a period.{contact_info}
        6. LANGUAGE: {language_instruction}
        7. NEVER mention other companies (Google, Flipkart, etc.) - ONLY Fasc Ai services.
        8. Use the context provided above to give accurate, specific information.
        
        Remember: You represent Fasc Ai EXCLUSIVELY. Keep it SHORT (1-2 sentences), COMPLETE, and NEVER ask questions.
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-5:])  # Last 5 messages for context
        
        # Add user intent message
        user_message = f"I want to know more. Please provide detailed information."
        messages.append({"role": "user", "content": user_message})
        
        # Generate response using Groq API (async)
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.5
        }
        
        response = None
        active_key = None
        attempts = 0
        
        async with httpx.AsyncClient(timeout=10.0) as client:  # Optimized from 30s to 15s for faster response
            while attempts < len(GROQ_API_KEYS):
                active_key = await _get_active_groq_key()
                headers = {
                    "Authorization": f"Bearer {active_key}",
                    "Content-Type": "application/json"
                }

                try:
                    response = await client.post(GROQ_API_URL, headers=headers, json=data)
                except httpx.RequestError as exc:
                    logger.error(f"Groq API request error with key {_mask_api_key(active_key)}: {exc}")
                    attempts += 1
                    await _advance_groq_key()
                    continue

                if response.status_code == 200:
                    break

                if response.status_code in (401, 403, 429):
                    logger.warning(
                        f"Groq API key {_mask_api_key(active_key)} returned status {response.status_code}. Rotating to next key."
                    )
                    attempts += 1
                    await _advance_groq_key()
                    continue

                logger.error(
                    f"Groq API error {response.status_code} with key {_mask_api_key(active_key)}: {response.text}"
                )
                raise HTTPException(
                    status_code=500,
                    detail="AI service temporarily unavailable. Please visit https://fascai.com for more information."
                )
        
        if not response or response.status_code != 200:
            logger.error("All configured Groq API keys have been exhausted or failed.")
            raise HTTPException(
                status_code=503,
                detail="AI service temporarily unavailable. Please visit https://fascai.com for more information."
            )
        
        if response.status_code == 200:
            result = response.json()
            detailed_reply = result['choices'][0]['message']['content'].strip()
            detailed_reply = strip_markdown(detailed_reply)
            logger.info(f"Generated detailed {followup_type} response via RAG + Groq")
            return detailed_reply
        else:
            raise HTTPException(
                status_code=500,
                detail="AI service temporarily unavailable. Please visit https://fascai.com for more information."
            )
            
    except Exception as e:
        logger.error(f"Error generating RAG-based response for {followup_type}: {e}")
        return None
async def get_contextual_response(context: Dict[str, Any], user_language: str = 'english', 
                                   conversation_history: List[Dict[str, str]] = None, 
                                   session_id: str = None) -> str:
    """
    Generate appropriate response based on conversation context with universal RAG support for ALL follow-up types
    """
    response_type = context.get("type", "general_followup")
    user_response = context.get("user_response", "")
    last_bot_response = context.get("last_bot_response", "")
    detected_service = context.get("detected_service")
    service_confidence = context.get("service_confidence", 0.0)
    
    # Only process "yes" responses - "no" responses get fallback
    if user_response.lower() not in ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'absolutely', 'definitely']:
        # Handle "no" responses
        if response_type == "pricing_followup":
            return "No problem! If you change your mind about pricing, feel free to ask anytime."
        elif response_type == "project_followup":
            return "No worries! Feel free to reach out when you're ready to start a project with us."
        elif response_type == "ai_benefits_followup":
            return "No problem! Is there anything else I can help you with regarding our services?"
        elif response_type == "service_info_followup":
            return "No worries! Feel free to ask if you need information about any of our services."
        else:
            return "No problem! If you have questions, feel free to ask anytime."
    
    # Universal RAG + Groq approach for ALL follow-up types
    # Generate queries based on follow-up type
    queries = generate_rag_query_for_followup(response_type, last_bot_response, detected_service)
    
    # Try to generate RAG-based response
    rag_response = await generate_rag_based_response(
        followup_type=response_type,
        queries=queries,
        last_bot_response=last_bot_response,
        user_response=user_response,
        conversation_history=conversation_history or [],
        user_language=user_language,
        detected_service=detected_service if response_type == "service_followup" else None
    )
    
    # If RAG-based response generated successfully, return it
    if rag_response:
        return rag_response
    
    # Fallback to existing responses if RAG fails
    logger.info(f"RAG-based response failed for {response_type}, using fallback response")
    
    # Fallback responses (existing logic)
    if response_type == "ai_benefits_followup":
        if user_language == 'hindi':
            return """AI आपके business को कई तरीकों से benefit कर सकता है:

🤖 Automation: Repetitive tasks को automate करके efficiency बढ़ाएं
📊 Data Analysis: Business data से valuable insights निकालें  
💬 Customer Support: 24/7 AI chatbots से customer service improve करें
🔍 Predictive Analytics: Future trends predict करके better decisions लें
🎯 Personalization: Customer experience को personalize करें
⚡ Cost Reduction: Manual work कम करके operational costs कम करें

हमारे successful AI implementations में Matson Surgicals (healthcare) और Grand Trio Sports (Kenya cricket services) शामिल हैं। 

आपके specific business needs के लिए AI solution के बारे में और जानना चाहेंगे?"""
        else:
            return """Great! AI can benefit your business in several key ways:

🤖 Automation: Automate repetitive tasks to increase efficiency
📊 Data Analysis: Extract valuable insights from your business data
💬 Customer Support: Improve customer service with 24/7 AI chatbots
🔍 Predictive Analytics: Predict future trends for better decision-making
🎯 Personalization: Enhance customer experience with personalized interactions
⚡ Cost Reduction: Reduce operational costs by minimizing manual work

Our successful AI implementations include Matson Surgicals (healthcare) and Grand Trio Sports (Kenya cricket services)."""
    
    elif response_type == "service_info_followup":
        return "Perfect! I can provide detailed information about our services including AI solutions, cloud computing, ERP, CRM, and more."
    
    elif response_type == "pricing_followup":
        return "Excellent! For detailed pricing information tailored to your specific needs, please email info@fascai.com or visit fascai.com/contact to speak with our sales team. They'll provide you with a customized quote based on your requirements."
    
    elif response_type == "project_followup":
        return "Fantastic! We'd love to collaborate with you on your project."
    
    elif response_type == "service_followup" and detected_service:
        # Fallback for service follow-up
        return f"I'd be happy to tell you more about our {detected_service.upper()} services! Our {detected_service.upper()} solutions are designed to help businesses streamline operations and improve efficiency."
    
    # Default response for general follow-ups
    return "I'm here to help you with Fasc Ai's services."

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None  # Add context for conversation flow
    
    class Config:
        # Validate message length and content
        max_length = 1000
        min_length = 1
    
    @validator('message')
    def validate_message(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v.strip()) > 1000:
            raise ValueError('Message too long (max 1000 characters)')
        # Basic XSS protection - remove potentially dangerous characters
        dangerous_chars = ['<script', '</script', 'javascript:', 'onload=', 'onerror=']
        v_lower = v.lower()
        for char in dangerous_chars:
            if char in v_lower:
                raise ValueError('Message contains potentially dangerous content')
        return v.strip()
    
    @validator('session_id')
    def validate_session_id(cls, v):
        if v and len(v) > 100:
            raise ValueError('Session ID too long')
        return v
class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = []
    website_url: str = "https://fascai.com"
    buttons: List[Dict[str, str]] = []  # Add buttons support for interactive workflows
    context: Optional[Dict[str, Any]] = None  # Add context for conversation flow
    
    @validator('reply', pre=True, always=True)
    def sanitize_reply(cls, v):
        return sanitize_response_text(v)
    
    class Config:
        # Ensure all fields are included in JSON response
        fields = {
            'reply': {'exclude': False},
            'sources': {'exclude': False},
            'website_url': {'exclude': False},
            'buttons': {'exclude': False},
            'context': {'exclude': False}
        }

class URLList(BaseModel):
    urls: List[str]
# Website Scraping Functions
def scrape_fascai_website(query: str) -> Dict[str, Any]:
    """
    Scrape fascai.com website for relevant information based on query
    Uses the new automatic crawling logic from rag_helper
    Returns structured data that can be added to ChromaDB
    """
    try:
        logger.info(f"Scraping fascai.com for query: {query}")
        
        # Main website URL
        base_url = "https://fascai.com"
        
        # Use the new automatic crawling logic from rag_helper
        from utils.rag_helper import scrape_website
        
        # Scrape all internal pages automatically (max depth 2)
        all_content = scrape_website(base_url, max_depth=2)
        
        if not all_content:
            logger.warning("No content scraped from fascai.com")
            return {}
        
        # Parse the scraped content and structure it
        # Split into sentences/chunks
        sentences = all_content.split('. ')
        
        scraped_content = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 20:  # Only meaningful chunks
                scraped_content.append({
                    'text': sentence.strip(),
                    'url': base_url,  # All content is from various pages
                    'type': 'content'
                })
        
        # Process and structure the scraped data
        structured_data = process_scraped_content(scraped_content, query)
        
        logger.info(f"Successfully scraped {len(structured_data)} content chunks from multiple pages on fascai.com")
        return structured_data
        
    except Exception as e:
        logger.error(f"Error scraping fascai.com: {str(e)}")
        return {}

def extract_page_content(soup: BeautifulSoup, url: str, query: str) -> List[Dict[str, str]]:
    """Extract relevant content from a webpage"""
    content_chunks = []
    
    try:
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text from different sections
        sections = [
            ('h1', 'heading'),
            ('h2', 'heading'), 
            ('h3', 'heading'),
            ('p', 'paragraph'),
            ('div', 'content'),
            ('section', 'section'),
            ('article', 'article')
        ]
        
        for tag, content_type in sections:
            elements = soup.find_all(tag)
            for element in elements:
                text = element.get_text().strip()
                if text and len(text) > 20:  # Only meaningful content
                    # Check if content is relevant to query
                    if is_content_relevant(text, query):
                        content_chunks.append({
                            'text': text,
                            'type': content_type,
                            'url': url,
                            'source': 'fascai_website'
                        })
        
        return content_chunks
        
    except Exception as e:
        logger.warning(f"Error extracting content from {url}: {str(e)}")
        return []

def is_content_relevant(text: str, query: str) -> bool:
    """Check if scraped content is relevant to the user query"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Keywords that indicate relevance
    relevant_keywords = [
        'project', 'portfolio', 'service', 'solution', 'technology',
        'development', 'design', 'implementation', 'client', 'customer',
        'experience', 'expertise', 'team', 'company', 'about', 'contact',
        'cloud', 'erp', 'crm', 'ai', 'iot', 'web', 'mobile', 'app'
    ]
    
    # Check if query keywords appear in text
    query_words = query_lower.split()
    for word in query_words:
        if word in text_lower and len(word) > 3:
            return True
    
    # Check if relevant keywords appear
    for keyword in relevant_keywords:
        if keyword in text_lower:
            return True
    
    return False
def process_scraped_content(content_chunks: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    """Process and structure scraped content for ChromaDB"""
    processed_chunks = []
    
    try:
        for i, chunk in enumerate(content_chunks):
            # Clean and structure the content
            clean_text = re.sub(r'\s+', ' ', chunk['text']).strip()
            
            if len(clean_text) > 50:  # Only meaningful chunks
                processed_chunks.append({
                    'id': f"scraped_content_{i}_{hash(clean_text) % 10000}",
                    'text': clean_text,
                    'metadata': {
                        'source': 'fascai_website',
                        'url': chunk['url'],
                        'type': chunk['type'],
                        'query': query,
                        'scraped_at': datetime.now().isoformat()
                    }
                })
        
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Error processing scraped content: {str(e)}")
        return []

def generate_response_from_scraped_data(scraped_data: List[Dict[str, str]], query: str) -> str:
    """Generate a response using scraped data when ChromaDB/Groq fails"""
    try:
        if not scraped_data:
            return "I'd be happy to help you with Fasc Ai's comprehensive IT solutions! We provide cloud hosting, ERP systems, CRM platforms, AI implementations, IoT solutions, custom web development, digital marketing, and digital transformation services. What specific area interests you?"
        
        # Extract relevant content from scraped data
        relevant_content = []
        for chunk in scraped_data[:5]:  # Use top 5 most relevant chunks
            if chunk.get('text'):
                relevant_content.append(chunk['text'])
        
        if not relevant_content:
            return "I'd be happy to help you with Fasc Ai's comprehensive IT solutions! We provide cloud hosting, ERP systems, CRM platforms, AI implementations, IoT solutions, custom web development, digital marketing, and digital transformation services. What specific area interests you?"
        
        # Create a simple response based on scraped content
        content_text = " ".join(relevant_content)
        
        # Simple keyword-based response generation
        if "project" in query.lower() or "projects" in query.lower():
            if "completed" in content_text.lower() or "delivered" in content_text.lower():
                return "Fasc Ai has successfully completed numerous projects across various domains including cloud computing, ERP implementations, CRM systems, AI solutions, IoT applications, web development, and digital transformation initiatives. Our experienced team has delivered solutions for clients across different industries."
            else:
                return "Fasc Ai has extensive experience in delivering successful projects across cloud computing, ERP systems, CRM platforms, AI implementations, IoT solutions, web development, and digital transformation. Our team has worked on diverse projects for clients in various industries."
        
        # Generic response for other queries
        return f"Based on our website content, Fasc Ai offers comprehensive IT solutions including cloud computing, ERP systems, CRM platforms, AI implementations, IoT solutions, web development, and digital transformation services. {content_text[:200]}..."
        
    except Exception as e:
        logger.error(f"Error generating response from scraped data: {str(e)}")
        return "I'd be happy to help you with Fasc Ai's comprehensive IT solutions! We provide cloud hosting, ERP systems, CRM platforms, AI implementations, IoT solutions, custom web development, digital marketing, and digital transformation services."

async def scheduled_scraping_job():
    """
    Scheduled job function that runs daily at 3 AM to scrape website and update ChromaDB
    This is a background task that automatically keeps the database up to date
    """
    try:
        logger.info("Starting scheduled daily scraping job at 3 AM")
        
        # Trigger scraping with a special query
        scraped_data = await asyncio.to_thread(scrape_fascai_website, "scheduled_daily_update")
        
        if scraped_data and len(scraped_data) > 0:
            # Add scraped content to ChromaDB
            success = await asyncio.to_thread(add_scraped_content_to_chromadb, scraped_data)
            
            if success:
                logger.info("Scheduled scraping job completed successfully")
            else:
                logger.warning("Scheduled scraping job completed but encountered issues adding to ChromaDB")
        else:
            logger.warning("Scheduled scraping job found no new content")

        # Ensure all priority queries remain preloaded/fresh
        for query in PRIORITY_PRELOAD_QUERIES:
            status, count = await _ensure_priority_query_preloaded(query)
            if status == "cached":
                logger.info(f"[Scheduled Job] Priority query cached: '{query}' ({count} relevant chunks)")
            elif status == "added":
                logger.info(f"[Scheduled Job] Priority query refreshed: '{query}' ({count} chunks scraped)")
            elif status == "empty":
                logger.warning(f"[Scheduled Job] No content found while refreshing query '{query}'")
            elif status == "add_failed":
                logger.warning(f"[Scheduled Job] Failed to update priority query '{query}' ({count} chunks scraped)")
            else:
                logger.warning(f"[Scheduled Job] Priority preload status '{status}' for query '{query}'")
            
    except Exception as e:
        logger.error(f"Error in scheduled scraping job: {str(e)}")

async def _ensure_priority_query_preloaded(query: str) -> Tuple[str, int]:
    """
    Ensure a specific query has indexed content in ChromaDB.
    Returns (status, count) for logging.
    """
    try:
        existing_results = await asyncio.to_thread(search_chroma, query, COLLECTION_NAME, 3)
        if existing_results:
            return ("cached", len(existing_results))
        
        scraped_data = await asyncio.to_thread(scrape_fascai_website, query)
        if not scraped_data:
            return ("empty", 0)
        
        added = await asyncio.to_thread(add_scraped_content_to_chromadb, scraped_data)
        if added:
            return ("added", len(scraped_data))
        return ("add_failed", len(scraped_data))
    except Exception as e:
        logger.error(f"Error preloading priority query '{query}': {str(e)}")
        return ("error", 0)

async def preload_priority_content():
    """
    Pre-ingest high-priority queries so responses are instant without live scraping.
    Runs in background on startup to keep accuracy while reducing latency.
    """
    logger.info("Starting priority content preload for latency-critical queries")
    
    for query in PRIORITY_PRELOAD_QUERIES:
        status, count = await _ensure_priority_query_preloaded(query)
        if status == "cached":
            logger.info(f"Priority query already covered: '{query}' ({count} relevant chunks)")
        elif status == "added":
            logger.info(f"Priority content ingested for query '{query}' ({count} chunks scraped)")
        elif status == "empty":
            logger.warning(f"Priority scraping produced no content for query '{query}'")
        elif status == "add_failed":
            logger.warning(f"Failed to add scraped priority content for query '{query}' ({count} chunks scraped)")
        else:
            logger.warning(f"Priority preload encountered status '{status}' for query '{query}'")
    
    logger.info("Priority content preload complete")

def add_scraped_content_to_chromadb(scraped_data: List[Dict[str, str]]) -> bool:
    """Add scraped content to ChromaDB for future use"""
    try:
        if not scraped_data:
            return False
        
        client = get_chroma_client()
        model = get_embedding_model()
        
        if not client or not model:
            return False
        
        collection = client.get_collection(COLLECTION_NAME)
        
        # Prepare data for ChromaDB
        texts = [chunk['text'] for chunk in scraped_data]
        ids = [chunk['id'] for chunk in scraped_data]
        metadatas = [chunk['metadata'] for chunk in scraped_data]
        
        # Check for existing IDs to avoid duplicates
        try:
            existing_results = collection.get(ids=ids)
            existing_ids = set(existing_results['ids']) if existing_results['ids'] else set()
            
            # Filter out existing IDs
            new_texts = []
            new_ids = []
            new_metadatas = []
            new_indices = []
            
            for i, chunk_id in enumerate(ids):
                if chunk_id not in existing_ids:
                    new_texts.append(texts[i])
                    new_ids.append(chunk_id)
                    new_metadatas.append(metadatas[i])
                    new_indices.append(i)
            
            # Only add if there are new items
            if new_texts:
                # Generate embeddings for new content only
                embeddings = model.encode(new_texts)
                
                # Add to collection
                collection.add(
                    embeddings=embeddings.tolist(),
                    documents=new_texts,
                    ids=new_ids,
                    metadatas=new_metadatas
                )
                
                logger.info(f"Added {len(new_texts)} new scraped content chunks to ChromaDB")
            else:
                logger.info("All scraped content already exists in ChromaDB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking existing IDs in ChromaDB: {str(e)}")
            return False
        
    except Exception as e:
        logger.error(f"Error adding scraped content to ChromaDB: {str(e)}")
        return False

# Initialize ChromaDB
def get_chroma_client():
    """Get cached ChromaDB client or initialize if not exists"""
    global chroma_client
    if chroma_client is None:
        try:
            chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDB client initialized and cached")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            return None
    return chroma_client

# Initialize embedding model
def get_embedding_model():
    """Get cached embedding model or initialize if not exists"""
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info("Embedding model initialized and cached")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            return None
    return embedding_model

# Web scraping functions
def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    # Remove very short words
    text = ' '.join([word for word in text.split() if len(word) > 2])
    return text.strip()
def extract_content(url: str) -> Dict[str, Any]:
    """Extract and clean content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        logger.info(f"Scraping website: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract main content
        title = soup.find('title')
        title_text = title.get_text().strip() if title else "No Title"
        
        # Extract text from different elements
        text_elements = []
        
        # Get headings
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text_elements.append(tag.get_text().strip())
        
        # Get paragraphs
        for tag in soup.find_all('p'):
            text_elements.append(tag.get_text().strip())
        
        # Get list items
        for tag in soup.find_all('li'):
            text_elements.append(tag.get_text().strip())
        
        # Get div content with substantial text
        for tag in soup.find_all('div'):
            text = tag.get_text().strip()
            if len(text) > 50:
                text_elements.append(text)
        
        # Clean and join text
        cleaned_text = clean_text(' '.join(text_elements))
        
        logger.info(f"Successfully scraped {len(cleaned_text)} characters from {url}")
        
        return {
            'url': url,
            'title': title_text,
            'content': cleaned_text,
            'timestamp': datetime.now().isoformat(),
            'content_length': len(cleaned_text)
        }
        
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {str(e)}")
        return None

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Only keep substantial chunks
            chunks.append(chunk.strip())
    
    return chunks
# Vector database functions
def store_content_in_chroma(urls: List[str], collection_name: str = COLLECTION_NAME) -> Dict[str, Any]:
    """Store scraped content in ChromaDB"""
    try:
        client = get_chroma_client()
        model = get_embedding_model()
        
        if not client or not model:
            raise Exception("Failed to initialize ChromaDB or embedding model")
        
        # Get or create collection
        try:
            collection = client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            collection = client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")
        
        total_chunks = 0
        successful_urls = 0
        failed_urls = []
        
        for url in urls[:MAX_URLS]:  # Limit to prevent overload
            try:
                logger.info(f"Processing URL: {url}")
                
                # Extract content
                content_data = extract_content(url)
                if not content_data or not content_data['content']:
                    failed_urls.append(url)
                    continue
                
                # Chunk the content
                chunks = chunk_text(content_data['content'])
                if not chunks:
                    failed_urls.append(url)
                    continue
                
                # Generate embeddings
                embeddings = model.encode(chunks)
                
                # Create unique IDs and metadata
                chunk_ids = []
                metadatas = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{hashlib.md5(url.encode()).hexdigest()[:8]}_chunk_{i}"
                    chunk_ids.append(chunk_id)
                    metadatas.append({
                        "url": url,
                        "title": content_data['title'],
                        "chunk_index": i,
                        "timestamp": content_data['timestamp'],
                        "content_length": content_data['content_length']
                    })
                
                # Check for existing IDs to avoid duplicates
                try:
                    existing_results = collection.get(ids=chunk_ids)
                    existing_ids = set(existing_results['ids']) if existing_results['ids'] else set()
                    
                    # Filter out existing IDs
                    new_chunks = []
                    new_ids = []
                    new_metadatas = []
                    new_embeddings = []
                    
                    for i, chunk_id in enumerate(chunk_ids):
                        if chunk_id not in existing_ids:
                            new_chunks.append(chunks[i])
                            new_ids.append(chunk_id)
                            new_metadatas.append(metadatas[i])
                            new_embeddings.append(embeddings[i])
                    
                    # Only add if there are new items
                    if new_chunks:
                        collection.add(
                            embeddings=new_embeddings.tolist(),
                            documents=new_chunks,
                            ids=new_ids,
                            metadatas=new_metadatas
                        )
                        logger.info(f"Stored {len(new_chunks)} new chunks from {url} (skipped {len(chunks) - len(new_chunks)} duplicates)")
                    else:
                        logger.info(f"All chunks from {url} already exist in ChromaDB")
                        
                except Exception as e:
                    logger.error(f"Error checking existing IDs for {url}: {str(e)}")
                    # Fallback to original behavior if checking fails
                    collection.add(
                        embeddings=embeddings.tolist(),
                        documents=chunks,
                        ids=chunk_ids,
                        metadatas=metadatas
                    )
                    logger.info(f"Stored {len(chunks)} chunks from {url}")
                
                total_chunks += len(chunks)
                successful_urls += 1
                
            except Exception as e:
                logger.error(f"Error processing {url}: {str(e)}")
                failed_urls.append(url)
        
        return {
            'success': True,
            'total_chunks': total_chunks,
            'successful_urls': successful_urls,
            'failed_urls': failed_urls,
            'collection_name': collection_name
        }
        
    except Exception as e:
        logger.error(f"Error storing content in ChromaDB: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }
def search_chroma(query: str, collection_name: str = COLLECTION_NAME, n_results: int = 3) -> List[Dict[str, Any]]:
    """Search ChromaDB for relevant content with relevance filtering"""
    try:
        client = get_chroma_client()
        model = get_embedding_model()
        
        if not client or not model:
            raise Exception("Failed to initialize ChromaDB or embedding model")
        
        # Get collection
        try:
            collection = client.get_collection(name=collection_name)
        except:
            raise Exception(f"Collection '{collection_name}' not found")
        
        # Generate query embedding with cache optimization
        global embedding_cache
        query_lower = query.lower().strip()
        if query_lower in embedding_cache:
            query_embedding = embedding_cache[query_lower]
        else:
            query_embedding = model.encode([query])
            # Cache the embedding (limit cache size to prevent memory issues)
            if len(embedding_cache) < 2000:  # Max 2000 cached embeddings (increased for better performance)
                embedding_cache[query_lower] = query_embedding
            else:
                # Clear cache if it gets too large (simple LRU: clear all and start fresh)
                embedding_cache = {query_lower: query_embedding}
        
        # Search
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        # Format results with relevance filtering
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                
                # Filter by relevance threshold
                if distance <= RELEVANCE_THRESHOLD:
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'distance': distance
                    })
                    logger.info(f"Result {i+1}: distance={distance:.4f} (relevant)")
                else:
                    logger.info(f"Result {i+1}: distance={distance:.4f} (filtered out - not relevant)")
        
        logger.info(f"Found {len(formatted_results)} relevant documents for query: {query} (after filtering)")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Error searching ChromaDB: {str(e)}")
        return []

# API Endpoints

@app.post("/crawl-and-store")
async def crawl_and_store(urls: URLList):
    """Crawl URLs and store content in ChromaDB"""
    try:
        result = store_content_in_chroma(urls.urls, COLLECTION_NAME)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")  # Rate limit: 10 messages per minute per IP
async def chat(request: Request, chat_request: ChatRequest):
    """Chat endpoint with ChromaDB RAG and intelligent conversation handling"""
    try:
        # Session management
        session_id = chat_request.session_id or str(uuid.uuid4())
        
        # Initialize session if it doesn't exist
        if session_id not in conversation_sessions:
            conversation_sessions[session_id] = {
                "conversations": [],
                "user_name": None,  # Store user's name for personalization
                "last_bot_response": None,  # Store last bot response for context
                "conversation_context": None,  # Store conversation context
                "project_context": {}  # Initialize project context
            }
            logger.info(f"Created new session: {session_id}")
        
        # Detect language from current message (turn-by-turn, no session storage)
        detected_language = detect_language(chat_request.message)
        user_language = detected_language
        soft_negative = is_soft_negative_message(chat_request.message)
        
        # Check for follow-up responses to previous bot questions (HIGHEST PRIORITY)
        last_bot_response = conversation_sessions[session_id].get("last_bot_response", "")
        if last_bot_response and is_follow_up_response(chat_request.message, last_bot_response):
            logger.info(f"Detected follow-up response: {chat_request.message}")
            
            # Get conversation context
            follow_up_context = get_follow_up_context(last_bot_response, chat_request.message)
            
            # Get conversation history for context-aware response
            conversation_history = conversation_sessions[session_id]["conversations"][-10:]
            
            # Generate contextual reply with RAG support (async)
            contextual_reply = await get_contextual_response(
                follow_up_context, 
                user_language,
                conversation_history=conversation_history,
                session_id=session_id
            )
            
            # Update conversation context
            conversation_sessions[session_id]["conversation_context"] = follow_up_context
            
            # Store in conversation history
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": contextual_reply})
            conversation_sessions[session_id]["last_bot_response"] = contextual_reply
            
            return ChatResponse(
                reply=contextual_reply,
                sources=[],
                website_url="https://fascai.com",
                context=follow_up_context
            )
        
        # Get conversation history (last 10 messages to limit context)
        conversation_history = conversation_sessions[session_id]["conversations"][-10:]
        
        # Initialize search_results variable (will be used later for RAG)
        search_results = None
        
        # FAST PATTERN CHECKS FIRST (for simple queries to avoid slow HF model)
        # Check for simple greetings/bye first - these are fast and don't need HF
        if is_greeting(chat_request.message):
            logger.info(f"Detected greeting: {chat_request.message}")
            greeting_reply = get_greeting_response(user_language, chat_request.message)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": greeting_reply})
            conversation_sessions[session_id]["last_bot_response"] = greeting_reply
            return ChatResponse(
                reply=greeting_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        if is_goodbye(chat_request.message):
            logger.info(f"Detected goodbye: {chat_request.message}")
            goodbye_reply = get_goodbye_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": goodbye_reply})
            return ChatResponse(
                reply=goodbye_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        # Fast pattern checks - BEFORE HF to save 2-3 seconds
        # Check for capability questions (fast pattern check before HF)
        message_lower = chat_request.message.lower()
        ai_automation_query = is_ai_automation_query(chat_request.message)
        service_keywords = ['erp', 'crm', 'cloud', 'hosting', 'iot', 'ai', 'service', 'services', 'solution', 'solutions']
        has_service_keywords = any(keyword in message_lower for keyword in service_keywords)
        if ai_automation_query and search_results is None:
            search_results = search_chroma(chat_request.message, COLLECTION_NAME, n_results=3)
        
        if not has_service_keywords and is_capability_question(chat_request.message):
            logger.info(f"Detected generic capability question (no service keywords): {chat_request.message}")
            capability_reply = get_capability_response(user_language, session_id)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": capability_reply})
            conversation_sessions[session_id]["last_bot_response"] = capability_reply
            return ChatResponse(
                reply=capability_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Check for help requests (fast pattern check before HF)
        # Removed hardcoded help request handling - everything now goes through RAG flow
        # RAG provides intelligent, context-aware, human-like responses for all queries
        
        # Check for acknowledgments (fast pattern check before HF)
        if is_acknowledgment(chat_request.message):
            logger.info(f"Detected acknowledgment: {chat_request.message}")
            ack_reply = get_acknowledgment_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": ack_reply})
            return ChatResponse(
                reply=ack_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Check for contact queries (fast pattern check before HF)
        if is_contact_query(chat_request.message):
            logger.info(f"Detected contact query: {chat_request.message}")
            
            # Search for contact information in ChromaDB
            contact_results = search_chroma(chat_request.message, COLLECTION_NAME, n_results=3)
            
            # Build context from ChromaDB results
            context = ""
            sources = []
            if contact_results:
                logger.info(f"Using ChromaDB vector search results for context ({len(contact_results)} relevant chunks)")
                context = "\n\n".join([result.get('content', '') for result in contact_results])
                sources = list(set([result.get('metadata', {}).get('url', '') for result in contact_results if result.get('metadata', {}).get('url')]))
            
            # Build context section
            if context and context.strip():
                context_section = f"""
        CRITICAL: The context provided below contains specific information about Fasc Ai's contact details, office locations, and contact information from our knowledge base. 
        You MUST base your answer PRIMARILY on this context. Do NOT give generic responses when specific context is available.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about Fasc Ai's contact information, office location, email, phone number, or how to reach the company. Provide detailed, human-like information based on the context above. Include contact details like email (info@fascai.com), phone number (+91-9958755444), website (fascai.com/contact), and any location information available in the context.
        Use the context provided above to give accurate, specific information about contact details. CRITICAL: The correct phone number for Fasc Ai is +91-9958755444. Always use this number when mentioning phone contact.
        """
            else:
                context_section = f"""
        Note: Use your general knowledge about Fasc Ai's contact information to answer questions accurately.
        The user is asking about Fasc Ai's contact information, office location, email, phone number, or how to reach the company. Provide detailed, human-like information.
        CRITICAL: The correct phone number for Fasc Ai is +91-9958755444. Always use this number when mentioning phone contact. The correct email address for Fasc Ai is info@fascai.com. Always include this email when user asks for email or contact information.
        """
            
            # Generate response using Groq API with context
            language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
            
            system_prompt = f"""
{context_section}
        
        You are an AI assistant, a friendly and helpful AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

        CRITICAL RULES - YOU MUST FOLLOW:
        1. CONTEXT USAGE - CRITICAL: When context is provided above, you MUST use it to answer questions accurately. Do NOT give generic responses when context provides specific details.
        2. CONTACT INFORMATION - ABSOLUTE CRITICAL: The correct phone number for Fasc Ai is +91-9958755444. When mentioning phone contact, ALWAYS use this exact number. Do NOT make up or use any other phone numbers. NEVER use "+91 11 4567 8900" or any other number. ONLY use +91-9958755444. If you write any other phone number, you have FAILED. The correct email address for Fasc Ai is info@fascai.com. When user asks for email or email id, ALWAYS provide info@fascai.com. NEVER say "we don't have a public email" or similar. ALWAYS provide info@fascai.com when asked about email.
        3. CRITICAL: Keep it SHORT and COMPLETE - MUST be 2-3 sentences MAX with key information, but make it natural and human-like. Keep response within 150 tokens.
        4. NEVER ask follow-up questions. NEVER end responses with questions. Just provide the information directly and end with a period.
        5. Be friendly, warm, and conversational in tone while staying professional.
        6. Show enthusiasm when discussing Fasc Ai's contact information.
        7. LANGUAGE: {language_instruction}
        8. NEVER mention other companies (Google, Flipkart, etc.) - ONLY Fasc Ai services.
        9. IMPORTANT: At the end of your response, politely and naturally suggest that they can use the 'New Project' or 'Existing Project' form buttons below to share their contact details (email, phone number, location) and get personalized assistance from our team. Make it sound friendly and helpful, not pushy. For example: "To share your email or contact details and receive personalized assistance, please use the form options below."
        
        Remember: You represent Fasc Ai EXCLUSIVELY. Keep it SHORT and COMPLETE. ALWAYS use phone number +91-9958755444.
        """
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": chat_request.message})
            
            # Call Groq API
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        GROQ_API_URL,
                        headers={
                            "Authorization": f"Bearer {await _get_active_groq_key()}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": messages,
                            "max_tokens": 100,
                            "temperature": 0.7
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        contact_reply = result['choices'][0]['message']['content'].strip()
                        contact_reply = strip_markdown(contact_reply)
                    else:
                        logger.error(f"Groq API error: {response.status_code}")
                        # Fallback response
                        contact_reply = "You can reach our team at +91-9958755444, info@fascai.com, or visit fascai.com/contact for direct assistance."
            except Exception as e:
                logger.error(f"Error calling Groq API for contact query: {str(e)}")
                # Fallback response
                contact_reply = "You can reach our team at +91-9958755444, info@fascai.com, or visit fascai.com/contact for direct assistance."
            
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": contact_reply})
            conversation_sessions[session_id]["last_bot_response"] = contact_reply
            
            # Add buttons for project workflow
            buttons = [
                {"text": "New Project", "action": "new_project"},
                {"text": "Existing Project", "action": "existing_project"}
            ]
            
            return ChatResponse(
                reply=contact_reply,
                sources=sources if sources else [],
                website_url="https://fascai.com",
                buttons=buttons
            )
        
        # Check for pricing queries (fast pattern check before HF)
        if is_pricing_query(chat_request.message):
            logger.info(f"Detected pricing query: {chat_request.message}")
            
            # Search for pricing information in ChromaDB
            pricing_results = search_chroma(chat_request.message, COLLECTION_NAME, n_results=3)
            
            # Build context from ChromaDB results
            context = ""
            sources = []
            if pricing_results:
                logger.info(f"Using ChromaDB vector search results for context ({len(pricing_results)} relevant chunks)")
                context = "\n\n".join([result.get('content', '') for result in pricing_results])
                sources = list(set([result.get('metadata', {}).get('url', '') for result in pricing_results if result.get('metadata', {}).get('url')]))
            
            # Build context section
            if context and context.strip():
                context_section = f"""
        CRITICAL: The context provided below contains specific information about Fasc Ai's pricing, packages, and pricing information from our knowledge base. 
        You MUST base your answer PRIMARILY on this context. Do NOT give generic responses when specific context is available.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about Fasc Ai's pricing, costs, packages, or payment information. Provide detailed, human-like information based on the context above. Mention that pricing varies based on requirements and they can contact info@fascai.com or visit fascai.com/contact for custom quotes.
        Use the context provided above to give accurate, specific information about pricing.
        """
            else:
                context_section = f"""
        Note: Use your general knowledge about Fasc Ai's pricing to answer questions accurately.
        The user is asking about Fasc Ai's pricing, costs, packages, or payment information. Provide detailed, human-like information. Mention that pricing varies based on requirements and they can contact info@fascai.com or visit fascai.com/contact for custom quotes.
        """
            
            # Generate response using Groq API with context
            language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
            
            system_prompt = f"""
{context_section}
        
        You are an AI assistant, a friendly and helpful AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

        CRITICAL RULES - YOU MUST FOLLOW:
        1. CONTEXT USAGE - CRITICAL: When context is provided above, you MUST use it to answer questions accurately. Do NOT give generic responses when context provides specific details.
        2. CRITICAL: Keep it SHORT and COMPLETE - MUST be 2-3 sentences MAX with key information, but make it natural and human-like. Keep response within 150 tokens.
        3. NEVER ask follow-up questions. NEVER end responses with questions. Just provide the information directly and end with a period.
        4. Be friendly, warm, and conversational in tone while staying professional.
        5. Show enthusiasm when discussing Fasc Ai's pricing and packages.
        6. LANGUAGE: {language_instruction}
        7. NEVER mention other companies (Google, Flipkart, etc.) - ONLY Fasc Ai services.
        8. IMPORTANT: At the end of your response, politely and naturally suggest that they can use the 'New Project' or 'Existing Project' form buttons below to share their contact details and get a personalized quote or pricing information. Make it sound friendly and helpful, not pushy. For example: "To share your requirements and receive a personalized quote, please use the form options below."
        
        Remember: You represent Fasc Ai EXCLUSIVELY. Keep it SHORT and COMPLETE.
        """
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": chat_request.message})
            
            # Call Groq API
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        GROQ_API_URL,
                        headers={
                            "Authorization": f"Bearer {await _get_active_groq_key()}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": messages,
                            "max_tokens": 100,
                            "temperature": 0.7
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        pricing_reply = result['choices'][0]['message']['content'].strip()
                        pricing_reply = strip_markdown(pricing_reply)
                    else:
                        logger.error(f"Groq API error: {response.status_code}")
                        # Fallback response
                        pricing_reply = "Pricing varies based on your specific requirements. For a custom quote, please contact our team at info@fascai.com or visit fascai.com/contact."
            except Exception as e:
                logger.error(f"Error calling Groq API for pricing query: {str(e)}")
                # Fallback response
                pricing_reply = "Pricing varies based on your specific requirements. For a custom quote, please contact our team at info@fascai.com or visit fascai.com/contact."
            
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": pricing_reply})
            conversation_sessions[session_id]["last_bot_response"] = pricing_reply
            
            # Add buttons for project workflow
            buttons = [
                {"text": "New Project", "action": "new_project"},
                {"text": "Existing Project", "action": "existing_project"}
            ]
            
            return ChatResponse(
                reply=pricing_reply,
                sources=sources if sources else [],
                website_url="https://fascai.com",
                buttons=buttons
            )
        
        # Check for policy queries (fast pattern check before HF)
        if is_policy_query(chat_request.message):
            logger.info(f"Detected policy query: {chat_request.message}")
            
            # Search for policy information in ChromaDB
            policy_results = search_chroma(chat_request.message, COLLECTION_NAME, n_results=3)
            
            # Build context from ChromaDB results
            context = ""
            sources = []
            if policy_results:
                logger.info(f"Using ChromaDB vector search results for context ({len(policy_results)} relevant chunks)")
                context = "\n\n".join([result.get('content', '') for result in policy_results])
                sources = list(set([result.get('metadata', {}).get('url', '') for result in policy_results if result.get('metadata', {}).get('url')]))
            
            # Build context section
            if context and context.strip():
                context_section = f"""
        CRITICAL: The context provided below contains specific information about Fasc Ai's company policies, terms, conditions, privacy policy, and legal information from our knowledge base. 
        You MUST base your answer PRIMARILY on this context. Do NOT give generic responses when specific context is available.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: The user is asking about Fasc Ai's company policies, terms and conditions, privacy policy, refund policy, or legal information. Provide detailed, human-like information based on the context above. If specific policy details are not in context, mention they can contact info@fascai.com or visit fascai.com/contact for detailed policy information.
        Use the context provided above to give accurate, specific information about policies.
        """
            else:
                context_section = f"""
        Note: Use your general knowledge about Fasc Ai's policies to answer questions accurately.
        The user is asking about Fasc Ai's company policies, terms and conditions, privacy policy, refund policy, or legal information. Provide detailed, human-like information. Mention they can contact info@fascai.com or visit fascai.com/contact for detailed policy information.
        """
            
            # Generate response using Groq API with context
            language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
            
            system_prompt = f"""
{context_section}
        
        You are an AI assistant, a friendly and helpful AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

        CRITICAL RULES - YOU MUST FOLLOW:
        1. CONTEXT USAGE - CRITICAL: When context is provided above, you MUST use it to answer questions accurately. Do NOT give generic responses when context provides specific details.
        2. CRITICAL: Keep it SHORT and COMPLETE - MUST be 2-3 sentences MAX with key information, but make it natural and human-like. Keep response within 150 tokens.
        3. NEVER ask follow-up questions. NEVER end responses with questions. Just provide the information directly and end with a period.
        4. Be friendly, warm, and conversational in tone while staying professional.
        5. Show enthusiasm when discussing Fasc Ai's policies and terms.
        6. LANGUAGE: {language_instruction}
        7. NEVER mention other companies (Google, Flipkart, etc.) - ONLY Fasc Ai services.
        8. IMPORTANT: At the end of your response, politely and naturally suggest that they can use the 'New Project' or 'Existing Project' form buttons below to share their contact details and get detailed policy information. Make it sound friendly and helpful, not pushy. For example: "To get detailed policy information and personalized assistance, please use the form options below."
        
        Remember: You represent Fasc Ai EXCLUSIVELY. Keep it SHORT and COMPLETE.
        """
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": chat_request.message})
            
            # Call Groq API
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        GROQ_API_URL,
                        headers={
                            "Authorization": f"Bearer {await _get_active_groq_key()}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "llama-3.1-8b-instant",
                            "messages": messages,
                            "max_tokens": 100,
                            "temperature": 0.7
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        policy_reply = result['choices'][0]['message']['content'].strip()
                        policy_reply = strip_markdown(policy_reply)
                    else:
                        logger.error(f"Groq API error: {response.status_code}")
                        # Fallback response
                        policy_reply = "For detailed information about our company policies, terms and conditions, please contact our team at info@fascai.com or visit fascai.com/contact."
            except Exception as e:
                logger.error(f"Error calling Groq API for policy query: {str(e)}")
                # Fallback response
                policy_reply = "For detailed information about our company policies, terms and conditions, please contact our team at info@fascai.com or visit fascai.com/contact."
            
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": policy_reply})
            conversation_sessions[session_id]["last_bot_response"] = policy_reply
            
            # Add buttons for project workflow
            buttons = [
                {"text": "New Project", "action": "new_project"},
                {"text": "Existing Project", "action": "existing_project"}
            ]
            
            return ChatResponse(
                reply=policy_reply,
                sources=sources if sources else [],
                website_url="https://fascai.com",
                buttons=buttons
            )
        # Check for bot identity (fast pattern check before HF)
        if is_bot_identity_question(chat_request.message):
            logger.info(f"Detected bot identity question: {chat_request.message}")
            identity_reply = get_bot_identity_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": identity_reply})
            conversation_sessions[session_id]["last_bot_response"] = identity_reply
            return ChatResponse(
                reply=identity_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        # Check for project management intent (if features are enabled) - ONLY when needed
        # Conditional call: Only when session context has waiting_for_existing_project OR query has project keywords
        if PROJECT_FEATURES_ENABLED:
            try:
                from project_manager import handle_project_workflow, is_project_intent
                # Get session context for project workflow
                session_context = conversation_sessions.get(session_id, {}).get('project_context', {})
                waiting_for_existing_project = session_context.get('waiting_for_existing_project', False)
                
                # Only call project workflow if:
                # 1. Session is waiting for existing project, OR
                # 2. Query has clear project intent keywords
                should_call_project_workflow = waiting_for_existing_project or is_project_intent(chat_request.message)
                
                if should_call_project_workflow:
                    logger.info(f"Calling handle_project_workflow with message: {chat_request.message} (session_context has waiting_for_existing_project: {waiting_for_existing_project}, has project intent: {is_project_intent(chat_request.message)})")
                    logger.info(f"Session context for {session_id}: {session_context}")
                    project_result = handle_project_workflow(chat_request.message, search_chroma, session_context)
                    logger.info(f"handle_project_workflow returned: {project_result}")
                    
                    if project_result:
                        logger.info(f"Detected project intent: {chat_request.message}")
                        logger.info(f"Project result: {project_result}")
                        conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
                        conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": project_result['reply']})
                        response = ChatResponse(
                            reply=project_result['reply'],
                            sources=project_result.get('sources', []),
                            website_url=project_result.get('website_url', "https://fascai.com"),
                            buttons=project_result.get('buttons', [])
                        )
                        logger.info(f"ChatResponse created: {response}")
                        return response
            except Exception as e:
                logger.error(f"Error in project workflow: {e}")
                # Continue with normal flow if project features fail
        
        if is_personality_question(chat_request.message):
            logger.info(f"Detected personality question: {chat_request.message}")
            personality_reply = get_personality_response(chat_request.message, user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": personality_reply})
            return ChatResponse(
                reply=personality_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Note: Bot identity, greeting, and goodbye checks are now handled early (before HF) for speed
        
        # Check if message is service inquiry (BEFORE acknowledgment to catch "ok tell me about services")
        if is_service_inquiry(chat_request.message):
            logger.info(f"Detected service inquiry: {chat_request.message}")
            
            # Detect specific service (AI, CRM, ERP, Cloud, IoT, Website, or None for generic)
            detected_service = detect_specific_service(chat_request.message)
            
            # Search for service information in ChromaDB
            service_results = search_chroma(chat_request.message, COLLECTION_NAME, n_results=3)
            
            # Build context from ChromaDB results
            context = ""
            sources = []
            if service_results:
                logger.info(f"Using ChromaDB vector search results for context ({len(service_results)} relevant chunks)")
                context = "\n\n".join([result.get('content', '') for result in service_results])
                sources = list(set([result.get('metadata', {}).get('url', '') for result in service_results if result.get('metadata', {}).get('url')]))
            
            # Build service-specific system prompt
            service_specific_instruction = ""
            if detected_service == 'ai':
                service_specific_instruction = "The user is asking specifically about Fasc Ai's AI services, artificial intelligence solutions, machine learning, and chatbot development. Focus ONLY on AI-related services and provide detailed information about AI implementations, chatbot development, and AI solutions."
            elif detected_service == 'crm':
                service_specific_instruction = "The user is asking specifically about Fasc Ai's CRM services and customer relationship management solutions. Focus ONLY on CRM-related services and provide detailed information about CRM implementation, customer management, and CRM solutions."
            elif detected_service == 'erp':
                service_specific_instruction = "The user is asking specifically about Fasc Ai's ERP services and enterprise resource planning systems. Focus ONLY on ERP-related services and provide detailed information about ERP implementation, business process management, and ERP solutions."
            elif detected_service == 'cloud':
                service_specific_instruction = "The user is asking specifically about Fasc Ai's cloud computing and cloud hosting services. Focus ONLY on cloud-related services and provide detailed information about cloud transformation, cloud hosting, and cloud solutions."
            elif detected_service == 'iot':
                service_specific_instruction = "The user is asking specifically about Fasc Ai's IoT services and Internet of Things solutions. Focus ONLY on IoT-related services and provide detailed information about IoT implementations and IoT solutions."
            elif detected_service == 'website':
                service_specific_instruction = "The user is asking specifically about Fasc Ai's website development and web design services. Focus ONLY on website-related services and provide detailed information about web development, web design, and e-commerce solutions."
            else:
                service_specific_instruction = "The user is asking about Fasc Ai's services in general. Provide a comprehensive overview of all IT solutions including cloud computing, ERP, CRM, AI solutions, IoT, and web development."
            
            # Build context section
            if context and context.strip():
                context_section = f"""
        CRITICAL: The context provided below contains specific information about Fasc Ai's services from our knowledge base. 
        You MUST base your answer PRIMARILY on this context. Do NOT give generic responses when specific context is available.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: {service_specific_instruction}
        Use the context provided above to give accurate, specific information about the services.
        """
            else:
                context_section = f"""
        Note: Use your general knowledge about Fasc Ai's services to answer questions accurately.
        {service_specific_instruction}
        """
            
            # Generate response using Groq API with context
            language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
            
            system_prompt = f"""
{context_section}
        
        You are an AI assistant, a friendly and helpful AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

        CRITICAL RULES - YOU MUST FOLLOW:
        1. CONTEXT USAGE - CRITICAL: When context is provided above, you MUST use it to answer questions accurately. Do NOT give generic "we help with X" responses when context provides specific details.
        2. CRITICAL: Keep it SHORT and COMPLETE - MUST be 2-3 sentences MAX with key information, but make it natural and human-like. Keep response within 150 tokens.
        3. NEVER ask follow-up questions. NEVER end responses with questions. Just provide the information directly and end with a period.
        4. Be friendly, warm, and conversational in tone while staying professional.
        5. Show enthusiasm when discussing Fasc Ai's services.
        6. LANGUAGE: {language_instruction}
        7. NEVER mention other companies (Google, Flipkart, etc.) - ONLY Fasc Ai services.
        
        Remember: You represent Fasc Ai EXCLUSIVELY. Keep it SHORT and COMPLETE.
        """
            
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)
            messages.append({"role": "user", "content": chat_request.message})
            
            # Generate response using Groq API (async)
            data = {
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.5
            }
            
            response = None
            active_key = None
            attempts = 0
            
            async with httpx.AsyncClient(timeout=10.0) as client:  # Optimized from 30s to 15s for faster response
                while attempts < len(GROQ_API_KEYS):
                    active_key = await _get_active_groq_key()
                    headers = {
                        "Authorization": f"Bearer {active_key}",
                        "Content-Type": "application/json"
                    }

                    try:
                        response = await client.post(GROQ_API_URL, headers=headers, json=data)
                    except httpx.RequestError as exc:
                        logger.error(f"Groq API request error with key {_mask_api_key(active_key)}: {exc}")
                        attempts += 1
                        await _advance_groq_key()
                        continue

                    if response.status_code == 200:
                        break

                    if response.status_code in (401, 403, 429):
                        logger.warning(
                            f"Groq API key {_mask_api_key(active_key)} returned status {response.status_code}. Rotating to next key."
                        )
                        attempts += 1
                        await _advance_groq_key()
                        continue

                    logger.error(
                        f"Groq API error {response.status_code} with key {_mask_api_key(active_key)}: {response.text}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="AI service temporarily unavailable. Please visit https://fascai.com for more information."
                    )

            if not response or response.status_code != 200:
                logger.error("All configured Groq API keys have been exhausted or failed.")
                raise HTTPException(
                    status_code=503,
                    detail="AI service temporarily unavailable. Please visit https://fascai.com for more information."
                )

            if response.status_code == 200:
                result = response.json()
                service_reply = result['choices'][0]['message']['content'].strip()
                service_reply = strip_markdown(service_reply)
            else:
                raise HTTPException(
                    status_code=500,
                    detail="AI service temporarily unavailable. Please visit https://fascai.com for more information."
                )
            
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": service_reply})
            conversation_sessions[session_id]["last_bot_response"] = service_reply
            
            return ChatResponse(
                reply=service_reply,
                sources=sources if sources else [],
                website_url="https://fascai.com"
            )
        
        # Note: Acknowledgment and goodbye checks are now handled early (before HF) for speed
        
        # Check if user is identifying as a client
        if is_client_identity(chat_request.message):
            logger.info(f"Detected client identity: {chat_request.message}")
            client_reply = get_client_identity_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": client_reply})
            return ChatResponse(
                reply=client_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Check if user is asking about projects
        if is_project_query(chat_request.message):
            logger.info(f"Detected project query: {chat_request.message}")
            project_reply = get_project_query_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": project_reply})
            return ChatResponse(
                reply=project_reply,
                sources=[],
                website_url="https://fascai.com/projects"
            )
        
        # Note: Capability questions are now handled early (before HF) for speed
        
        # Check if message is general help request (ONLY if no service keywords - let HF handle service-specific)
        if not has_service_keywords and is_general_help_request(chat_request.message):
            logger.info(f"Detected generic help request (no service keywords): {chat_request.message}")
            help_reply = get_general_help_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": help_reply})
            conversation_sessions[session_id]["last_bot_response"] = help_reply
            return ChatResponse(
                reply=help_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Check if message is meta/help question
        if is_meta_question(chat_request.message):
            logger.info(f"Detected meta question: {chat_request.message}")
            meta_reply = get_meta_response()
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": meta_reply})
            return ChatResponse(
                reply=meta_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Note: Contact queries are now handled early (before HF) for speed
        
        # Check if user is expressing dissatisfaction FIRST (highest priority)
        if is_dissatisfaction(chat_request.message):
            logger.info(f"Detected user dissatisfaction: {chat_request.message}")
            dissatisfaction_reply = get_dissatisfaction_response(user_language)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": dissatisfaction_reply})
            return ChatResponse(
                reply=dissatisfaction_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Check if user has a complaint or is dissatisfied
        if is_complaint(chat_request.message):
            logger.info(f"Detected complaint/dissatisfaction: {chat_request.message}")
            complaint_reply = get_complaint_response()
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": complaint_reply})
            return ChatResponse(
                reply=complaint_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Check if message is service information query (before pricing)
        if is_service_info_query(chat_request.message):
            logger.info(f"Detected service info query: {chat_request.message}")
            service_info_reply = get_service_info_response()
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": service_info_reply})
            conversation_sessions[session_id]["last_bot_response"] = service_info_reply
            return ChatResponse(
                reply=service_info_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Check if message is business info query (before pricing to avoid false positives)
        try:
            from project_manager import is_business_info_query, generate_business_info_response
            if is_business_info_query(chat_request.message):
                logger.info(f"Detected business info query: {chat_request.message}")
                business_info_result = generate_business_info_response(chat_request.message)
                conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
                conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": business_info_result['reply']})
                conversation_sessions[session_id]["last_bot_response"] = business_info_result['reply']
                return ChatResponse(
                    reply=business_info_result['reply'],
                    sources=business_info_result.get('sources', []),
                    website_url=business_info_result.get('website_url', "https://fascai.com"),
                    buttons=business_info_result.get('buttons', [])
                )
        except Exception as e:
            logger.error(f"Error in business info check: {e}")
        
        # NEW: Check for support queries (24/7 support, response time, etc.) - BEFORE RAG
        if is_support_query(chat_request.message):
            logger.info(f"Detected support query: {chat_request.message}")
            support_reply = await generate_support_response(chat_request.message, user_language, conversation_history)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": support_reply})
            conversation_sessions[session_id]["last_bot_response"] = support_reply
            return ChatResponse(
                reply=support_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # NEW: Check for specific project queries (dog walking project, funzoop project, etc.) - BEFORE RAG
        if is_specific_project_query(chat_request.message):
            logger.info(f"Detected specific project query: {chat_request.message}")
            project_reply = await generate_specific_project_response(chat_request.message, user_language, conversation_history)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": project_reply})
            conversation_sessions[session_id]["last_bot_response"] = project_reply
            return ChatResponse(
                reply=project_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # NEW: Check for company stats queries (how many projects, how many services, etc.) - BEFORE RAG
        if is_company_stats_query(chat_request.message):
            logger.info(f"Detected company stats query: {chat_request.message}")
            stats_reply = await generate_company_stats_response(chat_request.message, user_language, conversation_history)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": stats_reply})
            conversation_sessions[session_id]["last_bot_response"] = stats_reply
            return ChatResponse(
                reply=stats_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # NEW: Check for comparison queries (what makes you different, competitors, etc.) - BEFORE RAG
        if is_comparison_query(chat_request.message):
            logger.info(f"Detected comparison query: {chat_request.message}")
            comparison_reply = await generate_comparison_response(chat_request.message, user_language, conversation_history)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": comparison_reply})
            conversation_sessions[session_id]["last_bot_response"] = comparison_reply
            return ChatResponse(
                reply=comparison_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # NEW: Check for industry queries (what industries do you serve, etc.) - BEFORE RAG
        if is_industry_query(chat_request.message):
            logger.info(f"Detected industry query: {chat_request.message}")
            industry_reply = await generate_industry_response(chat_request.message, user_language, conversation_history)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": industry_reply})
            conversation_sessions[session_id]["last_bot_response"] = industry_reply
            return ChatResponse(
                reply=industry_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Note: Pricing queries are now handled early (before HF) for speed
        
        # Check if user is frustrated
        if is_frustrated(chat_request.message):
            logger.info(f"Detected frustrated user: {chat_request.message}")
            frustration_reply = get_frustration_response()
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": frustration_reply})
            return ChatResponse(
                reply=frustration_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Check if message is off-topic
        if is_off_topic(chat_request.message):
            logger.info(f"Detected off-topic query: {chat_request.message}")
            offtopic_reply = get_off_topic_response(chat_request.message)
            
            # Apply post-processing to remove any questions from off-topic responses
            import re
            # Remove question marks
            offtopic_reply = re.sub(r'[^.!?]*\?[^.!?]*', '', offtopic_reply)
            offtopic_reply = re.sub(r'\?+', '', offtopic_reply)
            # Remove question phrases
            question_phrase_patterns = [
                r'would you like to', r'do you want to', r'can i help you',
                r'what would you like', r'how can i assist', r'is there anything',
                r'are you exploring', r'what challenges', r'what\'s on your mind',
                r'how can we support', r'want to hear about', r'let me ask',
                r'since you\'re here', r'is there anything about'
            ]
            for phrase_pattern in question_phrase_patterns:
                offtopic_reply = re.sub(phrase_pattern + r'.*?[.!?]', '', offtopic_reply, flags=re.IGNORECASE)
            # Clean up
            offtopic_reply = re.sub(r'\s+', ' ', offtopic_reply).strip()
            # Ensure it ends with punctuation
            if offtopic_reply and not offtopic_reply.endswith(('.', '!')):
                offtopic_reply = offtopic_reply.rstrip() + '.'
            
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": offtopic_reply})
            return ChatResponse(
                reply=offtopic_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Search for relevant content (reuse early search if available for technical queries)
        if search_results is None:
            # Only search if we don't have early results
            search_results = search_chroma(chat_request.message, COLLECTION_NAME, n_results=3)
        else:
            logger.info(f"Reusing early ChromaDB search results: {len(search_results) if search_results else 0} results")
        # Prepare context and check relevance using distance-based detection
        context = ""  # Initialize context to empty string
        sources = []
        min_distance = 999.0  # Initialize with high value
        has_relevant_context = False
        
        if search_results:
            # Calculate minimum distance from search results
            distances = [result.get('distance', 999.0) for result in search_results]
            min_distance = min(distances) if distances else 999.0
            
            # Check if we have relevant context (distance <= RELEVANCE_THRESHOLD)
            has_relevant_context = min_distance <= RELEVANCE_THRESHOLD
            
            if has_relevant_context:
                context = " ".join([result['content'] for result in search_results])
                sources = list(set([result['metadata'].get('url', 'Unknown') for result in search_results]))
                logger.info(f"Using ChromaDB vector search results for context ({len(search_results)} relevant chunks, min_distance={min_distance:.4f})")
            else:
                logger.info(f"No relevant context found (min_distance={min_distance:.4f} > threshold={RELEVANCE_THRESHOLD}) - service likely not provided")
        else:
            # No search results at all
            logger.info("No ChromaDB search results found - attempting website scraping")
            min_distance = 999.0  # Set high distance to indicate no service found
            has_relevant_context = False
            
            # Try to scrape fascai.com for relevant information
            scraped_data = scrape_fascai_website(chat_request.message)
            
            if scraped_data:
                logger.info(f"Found {len(scraped_data)} relevant content chunks from website scraping")
                
                # Add scraped content to ChromaDB for future use
                add_scraped_content_to_chromadb(scraped_data)
                
                # Use scraped content to generate response (limit to prevent 413 error)
                max_chunks_for_groq = 5  # Only use top 5 most relevant chunks
                scraped_context = " ".join([chunk['text'] for chunk in scraped_data[:max_chunks_for_groq]])
                scraped_sources = list(set([chunk['metadata']['url'] for chunk in scraped_data[:max_chunks_for_groq]]))
                
                # Generate response using scraped content
                language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
                
                enhanced_system_prompt = f"""
                You are an AI assistant, a friendly and helpful AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

                CRITICAL RULES - YOU MUST FOLLOW:
                1. CRITICAL: Keep it SHORT and COMPLETE - MAXIMUM 2 SENTENCES ONLY - Keep responses short and concise (1-2 sentences) unless user asks for detailed explanations. Keep response within 150 tokens.
                2. ONLY answer questions about Fasc Ai's services, IT solutions, cloud computing, ERP, CRM, AI solutions, IoT, or related technology topics
                3. Use the scraped website content below to provide accurate, up-to-date information
                4. If the question is NOT about Fasc Ai or IT services, politely redirect to Fasc Ai's services
                5. Always maintain a professional yet friendly tone
                6. Include relevant website URLs when appropriate
                7. CLIENT CONTEXT: When asked about clients, mention MOF, Max Life, Lenovo, Medanta, Videocon, and Saarte. If someone says "I am your client", welcome them as a valued client WITHOUT asking which company they represent
                8. PROJECT CONTEXT: When asked about projects, list specific names: Grand Trio Sports, Wonder Land Garden, Funzoop, Tysley, Dorundo, Dog Walking, Matson Surgicals
                9. REMEMBER: Keep it SHORT and COMPLETE - Keep responses short and concise (1-2 sentences) unless user asks for detailed explanations
                10. LANGUAGE: {language_instruction}

                SCRAPED WEBSITE CONTENT:
                {scraped_context}

                User Query: {chat_request.message}
                """
                
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {await _get_active_groq_key()}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "llama-3.1-8b-instant",
                                "messages": [
                                    {"role": "system", "content": enhanced_system_prompt},
                                    {"role": "user", "content": chat_request.message}
                                ],
                                "temperature": 0.7,
                                "max_tokens": 100
                            },
                            timeout=10.0  # Optimized to 10s for faster response
                        )
                        
                        if response.status_code == 200:
                            groq_response = response.json()
                            ai_reply = groq_response['choices'][0]['message']['content']
                            # Strip markdown formatting from AI response
                            ai_reply = strip_markdown(ai_reply)
                            
                            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
                            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": ai_reply})
                            conversation_sessions[session_id]["last_bot_response"] = ai_reply
                            
                            return ChatResponse(
                                reply=ai_reply,
                                sources=scraped_sources,
                                website_url="https://fascai.com"
                            )
                        else:
                            logger.error(f"Groq API error: {response.status_code}")
                            # Use fallback response generation with scraped data
                            fallback_reply = generate_response_from_scraped_data(scraped_data, chat_request.message)
                            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
                            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": fallback_reply})
                            conversation_sessions[session_id]["last_bot_response"] = fallback_reply
                            return ChatResponse(
                                reply=fallback_reply,
                                sources=scraped_sources,
                                website_url="https://fascai.com"
                            )
                            
                except Exception as e:
                    logger.error(f"Error calling Groq API with scraped content: {str(e)}")
                    # Use fallback response generation with scraped data
                    fallback_reply = generate_response_from_scraped_data(scraped_data, chat_request.message)
                    conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
                    conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": fallback_reply})
                    conversation_sessions[session_id]["last_bot_response"] = fallback_reply
                    return ChatResponse(
                        reply=fallback_reply,
                        sources=scraped_sources,
                        website_url="https://fascai.com"
                    )
            
            # If scraping also fails, provide helpful fallback
            logger.info("Website scraping failed or no relevant content found - providing helpful fallback")
            import random
            fallback_responses = [
                "I'd be happy to help you with Fasc Ai's comprehensive IT solutions! We provide cloud hosting, ERP systems, CRM platforms, AI implementations, IoT solutions, custom web development, digital marketing, and digital transformation services.",
                "Fasc Ai offers complete IT solutions including cloud computing, ERP systems, CRM platforms, AI solutions, IoT implementations, web development, digital marketing, and business transformation services.",
                "Our comprehensive services include cloud hosting, ERP systems, CRM platforms, AI implementations, IoT solutions, custom web development, digital marketing, and enterprise transformation solutions.",
                "We specialize in cloud transformation, ERP/CRM deployments, AI-driven automation, and IoT platforms. Let me know which area you want to explore and I’ll share specific details.",
                "Our team delivers end-to-end technology solutions—AI strategy, automation, cloud migration, ERP/CRM rollouts, and IoT integration. Tell me your focus and I’ll guide you through how we can help."
            ]
            fallback_reply = random.choice(fallback_responses)
            conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
            conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": fallback_reply})
            conversation_sessions[session_id]["last_bot_response"] = fallback_reply
            return ChatResponse(
                reply=fallback_reply,
                sources=[],
                website_url="https://fascai.com"
            )
        
        # Create enhanced system prompt with strict domain adherence
        language_instruction = "You MUST respond in Hindi only." if detected_language == 'hindi' else "You MUST respond in English only."
        
        # Build context section based on availability and distance-based service detection
        # Extract user query for LLM analysis
        user_query = chat_request.message.strip()
        
        # NEW: Check if query is actually asking about a service we provide (not company info)
        # Queries like "software solutions", "custom software", "web development" are services we provide
        is_actual_service_query = False
        known_services_keywords = [
            'software solutions', 'custom software', 'software development',
            'web development', 'website development', 'app development',
            'cloud computing', 'erp', 'crm', 'ai solutions', 'iot',
            'automation', 'digital transformation', 'cloud hosting'
        ]
        user_query_lower = user_query.lower()
        
        # Check if query contains service keywords that we actually provide
        for service_keyword in known_services_keywords:
            if service_keyword in user_query_lower:
                is_actual_service_query = True
                logger.info(f"Query contains known service keyword '{service_keyword}' - treating as service query: {user_query}")
                break
        
        # Check if this is a definition query (what is, what are, explain)
        is_def_query = is_definition_query(user_query)
        
        # Check if query is asking about services we provide (vs asking if we provide a service)
        # Queries like "tell me about your software solutions" = service info query (we provide it)
        # Queries like "do you provide car servicing" = service query (we don't provide it)
        is_service_info_question = any(phrase in user_query_lower for phrase in [
            'tell me about your', 'tell me about', 'what are your', 'what is your',
            'your software', 'your services', 'your solutions', 'about your'
        ])
        
        # If query asks "tell me about your software solutions" - we DO provide software solutions
        if is_service_info_question and any(keyword in user_query_lower for keyword in ['software', 'solutions', 'services']):
            is_actual_service_query = True
            logger.info(f"Service info question detected - treating as service we provide: {user_query}")
        
        # Semantic verification: Check if service is actually provided
        # Skip service verification for definition queries - they should get informative responses
        # Also skip if ChromaDB confidence is very high (low distance < 0.3) - means very relevant context
        service_verified = False
        HIGH_CONFIDENCE_THRESHOLD = 0.3  # Very low distance = high confidence = skip verification
        
        if is_def_query:
            logger.info(f"Definition query detected, skipping service verification: {user_query}")
        elif has_relevant_context and min_distance < HIGH_CONFIDENCE_THRESHOLD:
            # High confidence from ChromaDB - skip verification to save time
            service_verified = True
            logger.info(f"High ChromaDB confidence (distance={min_distance:.4f} < {HIGH_CONFIDENCE_THRESHOLD}), skipping service verification for query: {user_query}")
        elif is_actual_service_query:
            # Query is about services we provide - verify it
            if has_relevant_context and context and context.strip():
                service_verified = verify_service_provided(user_query, context, search_results)
                logger.info(f"Service verification result for actual service query: {service_verified} for query: {user_query}")
                # If verification fails but we know we provide this service, still mark as verified
                if not service_verified and is_actual_service_query:
                    logger.info(f"Query is about known service we provide, marking as verified: {user_query}")
                    service_verified = True
            else:
                # No context but query is about services we provide - assume we provide it
                service_verified = True
                logger.info(f"No context but query is about known service, marking as verified: {user_query}")
        elif not is_def_query and has_relevant_context and context and context.strip():
            # Only verify if confidence is not very high and it's not a known service
            service_verified = verify_service_provided(user_query, context, search_results)
            logger.info(f"Service verification result: {service_verified} for query: {user_query}")
            if not service_verified and ai_automation_query:
                logger.info("AI/automation query detected with relevant context - relaxing verification requirement.")
                service_verified = True
        
        if is_def_query and has_relevant_context and context and context.strip():
            # Definition query with context - provide informative answer about the topic
            context_section = f"""
        QUERY TYPE: DEFINITION QUERY - User is asking "what is" or "explain" about a topic.
        User Query: "{user_query}"
        Context Found: Relevant information found in knowledge base (distance={min_distance:.4f} <= threshold={RELEVANCE_THRESHOLD}).
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS:
        - Provide a brief, informative definition/explanation (1-2 sentences max).
        - If the context mentions that Fasc Ai provides services related to this topic, mention that we provide these services.
        - Example: If query is "what is artificial intelligence" and context mentions AI services, say: "Artificial intelligence (AI) is... We provide AI solutions including..."
        - ALWAYS keep response short (1-2 sentences, 100 tokens max).
        - Be accurate and human-like in your response.
        - CRITICAL: Ensure your response is COMPLETE - end with a full sentence and period. Do NOT cut off mid-sentence.
        - If context mentions Fasc Ai provides related services, mention them naturally in your response.
"""
        elif has_relevant_context and context and context.strip() and service_verified:
            # Service verified in knowledge base - EXACT match found
            context_section = f"""
        CRITICAL: The context provided below contains specific information about Fasc Ai's services from our knowledge base. 
        You MUST base your answer PRIMARILY on this context. Do NOT give generic responses when specific context is available.
        
        SERVICE DETECTION ANALYSIS - CRITICAL:
        User Query: "{user_query}"
        Context Found: Relevant service found in knowledge base (distance={min_distance:.4f} <= threshold={RELEVANCE_THRESHOLD}).
        Service Verification: EXACT service match verified in context.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: 
        - Provide helpful, accurate information about the service (1-2 sentences max).
        - Use the context above to give specific details about the service.
        - ALWAYS keep response short (1-2 sentences, 100 tokens max).
        - Be accurate and human-like in your response.
        - CRITICAL: Ensure your response is COMPLETE - end with a full sentence and period. Do NOT cut off mid-sentence.
"""
        elif has_relevant_context and context and context.strip() and not service_verified:
            # Context found but service NOT verified - check if it's actually a service query
            if is_actual_service_query:
                # Query is about services we provide - provide information even if verification failed
                context_section = f"""
        CRITICAL: The context provided below contains information about Fasc Ai's services.
        User Query: "{user_query}"
        Context Found: Related content found in knowledge base (distance={min_distance:.4f} <= threshold={RELEVANCE_THRESHOLD}).
        
        NOTE: This query is about services we provide (software solutions, web development, etc.).
        Provide helpful information about these services based on the context.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: 
        - Provide helpful, accurate information about the service (1-2 sentences max).
        - Use the context above to give specific details about the service.
        - ALWAYS keep response short (1-2 sentences, 100 tokens max).
        - Be accurate and human-like in your response.
        - CRITICAL: Ensure your response is COMPLETE - end with a full sentence and period.
        """
            else:
                # Context found but service NOT verified - RELATED but NOT EXACT
                context_section = f"""
        SERVICE DETECTION ANALYSIS - CRITICAL:
        User Query: "{user_query}"
        Context Found: Related content found in knowledge base (distance={min_distance:.4f} <= threshold={RELEVANCE_THRESHOLD}).
        Service Verification: EXACT service match NOT verified - context contains related but NOT exact service.
        
        CRITICAL INSTRUCTIONS:
        - The context contains RELATED services but NOT the EXACT service mentioned in the query.
        - You MUST politely decline and redirect to our core services.
        - Example: Query "hardware installation" + Context "IT solutions" → NOT EXACT. Decline politely.
        - Example: Query "courier services" + Context "logistics" → NOT EXACT. Decline politely.
        
        TECHNICAL CONTEXT FROM COMPANY KNOWLEDGE BASE:
        {context}
        
        INSTRUCTIONS: 
        - Politely acknowledge that we don't provide this specific service.
        - Redirect to our core services: cloud computing, ERP, CRM, AI implementations, and IoT services.
        - Keep it short (1-2 sentences max, 100 tokens max) and friendly.
        - CRITICAL: Ensure your response is COMPLETE - end with a full sentence and period. Do NOT cut off mid-sentence.
        - Example response: "Thank you for your interest. While we don't provide [service name] as a dedicated service, we can help with cloud computing, ERP, CRM, AI implementations, and IoT services."
"""
        elif is_def_query and (not has_relevant_context or not context):
            # Definition query with no context - provide general definition
            context_section = f"""
        QUERY TYPE: DEFINITION QUERY - User is asking "what is" or "explain" about a topic.
        User Query: "{user_query}"
        Context Found: No relevant information found in knowledge base (distance={min_distance:.4f} > threshold={RELEVANCE_THRESHOLD}).
        
        INSTRUCTIONS:
        - Provide a brief, informative definition/explanation based on general knowledge (1-2 sentences max).
        - If the topic is related to IT/technology (AI, cloud, ERP, CRM, IoT, etc.), mention that Fasc Ai provides related services.
        - Example: If query is "what is artificial intelligence", say: "Artificial intelligence (AI) is... We provide AI solutions including machine learning, chatbots, and automation."
        - ALWAYS keep response short (1-2 sentences, 100 tokens max).
        - Be accurate and human-like in your response.
        - CRITICAL: Ensure your response is COMPLETE - end with a full sentence and period. Do NOT cut off mid-sentence.
"""
        elif not has_relevant_context or not context:
            # No relevant service found - check if it's about services we provide
            if is_actual_service_query:
                # Query is about services we provide but no context found - provide general info
                context_section = f"""
        QUERY TYPE: SERVICE INFORMATION QUERY - User is asking about services we provide.
        User Query: "{user_query}"
        Context Found: No relevant information found in knowledge base (distance={min_distance:.4f} > threshold={RELEVANCE_THRESHOLD}).
        
        NOTE: This query is about services we provide (software solutions, web development, etc.).
        Even though no specific context was found, provide information about these services.
        
        INSTRUCTIONS:
        - Provide helpful information about the service mentioned (1-2 sentences max).
        - Mention that we provide these services and give general information.
        - Example: If query is "software solutions", say: "We offer custom software development services as part of our enterprise solutions, empowering businesses to achieve their unique goals."
        - ALWAYS keep response short (1-2 sentences, 100 tokens max).
        - Be accurate and human-like in your response.
        - CRITICAL: Ensure your response is COMPLETE - end with a full sentence and period.
        """
            else:
                # No relevant service found - politely decline (only for actual service queries we don't provide)
                context_section = f"""
        SERVICE DETECTION: No relevant service found in knowledge base (distance={min_distance:.4f} > threshold={RELEVANCE_THRESHOLD}).
        
        CRITICAL QUERY ANALYSIS:
        - Analyze the user's query intent FIRST before responding.
        - If query is asking "do you provide X" or "can you help with X" where X is NOT a service we provide → politely decline.
        - If query is asking about company info, statistics, industries, or comparisons → provide information (DO NOT use denial template).
        - Only use denial template for actual service queries we don't provide.
        
        INSTRUCTIONS: 
        - FIRST: Analyze if this is a service query or company info query.
        - If service query we don't provide: Politely acknowledge that we don't provide this specific service and redirect to our core services.
        - If company info query: Provide helpful information about the company (projects, services, industries, etc.).
        - Keep it short (1-2 sentences max, 100 tokens max) and friendly.
        - CRITICAL: Ensure your response is COMPLETE - end with a full sentence and period.
        
        Example for service we don't provide: "Thank you for your interest. While we don't provide [service name] as a dedicated service, we can help with cloud computing, ERP, CRM, AI implementations, and IoT services."
        Example for company info: "We've completed 250+ projects including Grand Trio Sports, Funzoop, Wonder Land Garden, Tysley, Dorundo, Dog Walking, and Matson Surgicals."
        """
        else:
            context_section = """
        Note: Use your general knowledge about Fasc Ai's services to answer questions accurately.
"""
        
        system_prompt = f"""
{context_section}
        
        You are an AI assistant, a friendly and helpful AI assistant for Fasc Ai Ventures Private Limited IT solutions company.

        CRITICAL DOMAIN SAFETY RULES - ABSOLUTE PRIORITY:
        - YOU ONLY represent Fasc Ai Ventures Private Limited. This is NON-NEGOTIABLE.
        - NEVER mention, discuss, or provide information about Google, Flipkart, Amazon, Microsoft, or ANY other company's services.
        - If asked about other companies (e.g., "Tell me about Google Cloud" or "What services does Flipkart offer"), IMMEDIATELY redirect: "I'd be happy to help you with Fasc Ai's IT solutions instead. We offer cloud hosting, ERP, CRM, AI solutions, and IoT services."
        - ALL answers MUST be about FASC.AI services ONLY. No exceptions.
        
        CRITICAL QUERY INTENT ANALYSIS - NEW:
        Before responding, analyze the user's query intent:
        
        1. SERVICE QUERY: User is asking if you provide a specific service (e.g., "do you provide car servicing")
           → If service provided: Give info
           → If service NOT provided: Use denial template
        
        2. COMPANY INFO QUERY: User is asking about company statistics/info (e.g., "how many projects", "how many services")
           → Always answer with company information (250+ projects, services list, etc.)
           → DO NOT use denial template
        
        3. SERVICE INFO QUERY: User is asking about services we provide (e.g., "tell me about your software solutions")
           → Always provide information about the service
           → DO NOT use denial template
        
        4. INDUSTRY QUERY: User is asking about industries served
           → List industries we serve
           → DO NOT use denial template
        
        5. COMPARISON QUERY: User is asking about competitors/differences
           → Answer about company advantages/unique selling points
           → DO NOT use denial template
        
        CRITICAL RULES - YOU MUST FOLLOW:
        1. QUERY ANALYSIS & SERVICE DETECTION - CRITICAL: Follow the QUERY ANALYSIS INSTRUCTIONS above. Analyze the user's query and context carefully. 
           - If query is about services we provide (software solutions, web development, etc.) → provide information
           - If query is about company info (project count, services count, industries) → provide information
           - If query is asking "do you provide X" where X is NOT a service we provide → politely decline
           - Be intelligent and accurate in your analysis. DO NOT use denial template for company info queries.
        2. RESPONSE LENGTH - ABSOLUTE CRITICAL: You MUST keep responses SHORT. Maximum 1-2 sentences ONLY. Maximum 100 tokens. DO NOT exceed this limit. DO NOT write long paragraphs. DO NOT write 3+ sentences. If you write more than 2 sentences, you have FAILED. Count your sentences before responding. REMEMBER: 1-2 sentences MAX, 100 tokens MAX.
        2a. RESPONSE COMPLETENESS - ABSOLUTE CRITICAL: ALWAYS ensure your response is COMPLETE. End with a full sentence and period. Do NOT cut off mid-sentence. Do NOT end mid-word or mid-phrase. If you're approaching the token limit, STOP and complete your current sentence with a period before ending. A complete response ending with a period is more important than using all tokens. NEVER leave responses incomplete or hanging.
        2b. NO HARDCODED CONTENT - CRITICAL: Do NOT mention specific blog post titles, article names, or specific content titles from context. Keep responses generic and intelligent. If redirecting to services, just mention "IT solutions" or "our services" without listing specific blog posts or articles.
        2c. CONTACT NUMBER - ABSOLUTE CRITICAL: The ONLY correct phone number for Fasc Ai is +91-9958755444. NEVER use any other phone number. NEVER make up phone numbers. NEVER use "+91 11 4567 8900" or similar. If you write any phone number other than +91-9958755444, you have FAILED.
        3. CONTEXT USAGE - CRITICAL: When context is provided above, you MUST use it to answer questions accurately. Do NOT give generic "we help with X" responses when context provides specific details. If no context is provided, use general knowledge about Fasc Ai's services ONLY. Do NOT mention specific blog post names or article titles from context.
        4. DOMAIN RESTRICTION - ABSOLUTE: ONLY answer questions about Fasc Ai's services, IT solutions, cloud computing, ERP, CRM, AI solutions, IoT, or related technology topics. If question mentions other companies (Google, Flipkart, etc.), redirect to FASC.AI services.
        5. When asked about services, ALWAYS mention our core services: IT solutions, cloud computing, ERP, CRM, AI solutions, and IoT implementations
        6. If the question is NOT about Fasc Ai or IT services, politely and warmly redirect the user back to Fasc Ai's services. Acknowledge their message briefly if appropriate, then gently guide them to our services with a friendly tone. Keep it short (1-2 sentences). Do NOT mention specific blog posts or articles. Just say: "That's interesting! I'd be happy to help you with Fasc Ai's IT solutions instead. We offer cloud hosting, ERP, CRM, AI solutions, and IoT services."
        7. CRITICAL: NEVER ask follow-up questions. NEVER end responses with questions like "What would you like to know?", "What do you need help with?", "Would you like to know more?", "What are you interested in?", "What specific services are you looking for?", "Can I help you with something?", "How can I assist you?", "Is there anything else I can help you with?", "Would you like to know more about...?", etc. Just provide the information directly and end with a period. THIS IS ABSOLUTELY FORBIDDEN - NO EXCEPTIONS. If you write a question mark (?), you have FAILED. Your response MUST end with a period (.) only. NEVER write any sentence ending with a question mark - ALL questions will be removed automatically.
        
        CRITICAL - FORBIDDEN PHRASES: NEVER use these phrases in your responses:
        - "Would you like to" (any variation: "Would you like to know", "Would you like to learn", "Would you like to discuss", "Would you like to explore", "Would you like to tell me", "Would you like more", "Would you like additional", "Would you like to hear", "Would you like to find out")
        - "Do you want to" (any variation: "Do you want to know", "Do you want to learn", "Do you want to discuss", "Do you want to explore")
        - "Can I help you with" (any variation)
        - "What would you like to" (any variation: "What would you like", "What would you like to explore")
        - "How can I assist you" (any variation)
        - "How can we support" (any variation: "How can we support your business goals")
        - "Is there anything else" (any variation)
        - "Is there anything about" (any variation: "Is there anything about cloud services... I can help you with")
        - "Are you exploring" (any variation: "Are you exploring any technology upgrades")
        - "What challenges" (any variation: "What challenges is your business facing")
        - "What's on your mind" (any variation: "What's on your mind business-wise")
        - "Want to hear about" (any variation: "Want to hear about our services")
        - "Let me ask" (any variation: "Let me ask - are you exploring")
        - "Since you're here" (any variation: "Since you're here, is there anything")
        
        If you write ANY of these phrases, you have FAILED. NEVER use "Would you like to", "Do you want to", "Can I help you", "What would you like", "How can I assist", "How can we support", "Is there anything else", "Is there anything about", "Are you exploring", "What challenges", "What's on your mind", "Want to hear about", "Let me ask", "Since you're here" - these are ABSOLUTELY FORBIDDEN. Just provide information directly and end with a period.
        
        EXPLICIT EXAMPLES - FOLLOW THESE:
        WRONG: "Would you like to discuss how we can help you enhance your digital profile or skills?"
        CORRECT: "We can help you enhance your digital profile or skills with our IT solutions and services."
        
        WRONG: "Would you like to discuss how we can assist you with your digital transformation needs?"
        CORRECT: "We can assist you with your digital transformation needs through our cloud computing, ERP, CRM, AI solutions, and IoT services."
        
        WRONG: "Would you like to discuss further about your project requirements?"
        CORRECT: "We can help you with your project requirements using our AI-powered chatbot solutions."
        
        WRONG: "Can you please share more about your specific requirements?"
        CORRECT: "We provide cloud computing, ERP, CRM, AI solutions, and IoT services to meet your specific requirements."
        
        WRONG: "Would you like to explore our services in digital transformation, social media management, or online reputation building?"
        CORRECT: "We offer digital transformation services, cloud computing, ERP, CRM, AI solutions, and IoT services."
        
        WRONG: "Since you're here, is there anything about cloud services, digital transformation, or enterprise applications I can help you with?"
        CORRECT: "We offer cloud services, digital transformation, and enterprise applications."
        
        WRONG: "Let me ask - are you exploring any technology upgrades or digital transformation for your organization?"
        CORRECT: "We can help with technology upgrades and digital transformation for your organization."
        
        WRONG: "What's on your mind business-wise?"
        CORRECT: "We offer cloud computing, ERP, CRM, AI solutions, and IoT services for your business needs."
        
        WRONG: "What challenges is your business facing?"
        CORRECT: "We help businesses with cloud computing, ERP, CRM, AI solutions, and IoT services."
        
        WRONG: "How can we support your business goals?"
        CORRECT: "We support your business goals through cloud computing, ERP, CRM, AI solutions, and IoT services."
        
        WRONG: "What would you like to explore?"
        CORRECT: "We offer cloud computing, ERP, CRM, AI solutions, and IoT services."
        
        WRONG: "Would you like to know more about our services?"
        CORRECT: "We offer cloud computing, ERP, CRM, AI solutions, and IoT services."
        
        WRONG: "I'd be happy to help you with that, but it seems your question is not related to Fasc Ai's services or IT solutions. At Fasc Ai, we empower digital transformations for our esteemed clients, providing elite digital transformation services in areas like cloud, ERP, CRM, mobility, and IoT. If you're interested in improving your digital presence or learning more about our services, I'd be happy to help. Would you like to know more about our services?"
        CORRECT: "I'd be happy to help you with that, but it seems your question is not related to Fasc Ai's services or IT solutions. At Fasc Ai, we empower digital transformations for our esteemed clients, providing elite digital transformation services in areas like cloud, ERP, CRM, mobility, and IoT. We offer cloud computing, ERP, CRM, AI solutions, and IoT services."
        Remember: NEVER ask questions. ALWAYS provide information directly and end with a period. NEVER use "Would you like to explore", "What would you like to explore", "How can we support", "What challenges", "What's on your mind", "Let me ask", "Since you're here", "Would you like to know more about our services" - these are ABSOLUTELY FORBIDDEN.
        8. NEGATIVE ACKNOWLEDGMENT - CRITICAL: When user says they don't need help (e.g., "i don't need your help", "i don't need help", "i am not your client"), acknowledge politely and briefly (1-2 sentences MAX), without being pushy. Accept their decision gracefully. For example: "Understood. If you change your mind or have any questions about our services, feel free to reach out anytime." Keep it short, respectful, and end with a period.
        9. HELP REQUESTS - ABSOLUTE CRITICAL: When user asks for help (e.g., "i need help", "i need assistance", "i am facing problems"), provide intelligent, context-aware, helpful responses. If context exists, use it. If no context, offer general help with our services. Keep it short (1-2 sentences max). CRITICAL: NEVER ask questions in help responses. NEVER say "What do you need help with?" or similar. PROVIDE information directly, DO NOT ask questions. Example: "I'm here to help! We offer cloud computing, ERP, CRM, AI solutions, and IoT services. What specific area would you like to know about?" is WRONG. Correct: "I'm here to help! We offer cloud computing, ERP, CRM, AI solutions, and IoT services. Feel free to ask about any of these." or "I'm here to help! We offer cloud computing, ERP, CRM, AI solutions, and IoT services."
        10. Always end each sentence with a full stop (.)
        11. Give complete answers without cutting off mid-sentence - ABSOLUTE CRITICAL: If approaching token limit, STOP immediately and complete your current sentence with a period before ending. Do NOT continue writing if you're near the limit - finish the sentence you're on. A complete sentence ending with a period is more important than using all 100 tokens. Check if your response ends with a period - if not, you have FAILED.
        12. Be friendly, warm, and conversational in tone while staying professional
        13. Never use bullet points, lists, or formatting
        14. NEVER answer questions about other companies, jobs at other companies, personal topics, or unrelated subjects. ALWAYS redirect to FASC.AI services.
        15. DO NOT write long paragraphs - keep it brief and to the point. REMEMBER: 1-2 sentences MAX, 100 tokens MAX. Ensure responses are COMPLETE - complete sentences before ending.
        16. Show enthusiasm and helpfulness when discussing Fasc Ai's services
        17. IMPORTANT: When discussing services, always include "AI solutions" as one of our key offerings
        18. CLIENT CONTEXT DISAMBIGUATION - CRITICAL: When asked about "our clients" or "who are your clients", list our portfolio clients: MOF (Ministry of Finance), Max Life Insurance, Lenovo, Medanta, Videocon, and Saarte. These are for REFERENCE when users ask about our portfolio. NEVER ask someone who says "I am your client" which of these companies they represent. Treat anyone who identifies as "our client" or "existing client" as a SEPARATE client reaching out for support. CRITICAL: NEVER assume a user is a client unless they explicitly say "I am your client" or "I am your existing client". Treat all users as potential customers by default. Do NOT use phrases like "valued client" or "as a client" unless the user explicitly identifies as a client.
        19. PROJECT CONTEXT - CRITICAL: When asked about projects, ALWAYS mention specific project names: Grand Trio Sports (Kenya cricket services), Wonder Land Garden and Funzoop (Shopify eCommerce), Tysley (AI chat platform), Dorundo (scooter rental), Dog Walking (pet services), and Matson Surgicals (healthcare website). DO NOT give generic answers like "250+ projects" - LIST ACTUAL PROJECT NAMES.
        20. LANGUAGE: {language_instruction}
        
        FINAL REMINDER: Response length and completeness are ABSOLUTE CRITICAL. 1-2 sentences MAX. 100 tokens MAX. Count your sentences. If you write more, you have FAILED. Ensure your response is COMPLETE - MUST end with a full sentence and period. Do NOT cut off mid-sentence. Do NOT end mid-word. Check your response ends with a period before sending. NEVER mention specific blog post names or article titles - keep it generic. NEVER ask questions in help responses - PROVIDE information directly. NEVER end with a question mark (?) - ALWAYS end with a period (.). If your response contains a question mark, you have FAILED. NEVER use any phone number other than +91-9958755444 - if you write any other number, you have FAILED. NEVER use forbidden phrases like "Would you like to", "Do you want to", "Can I help you", "What would you like", "How can I assist", "Is there anything else" - if you write ANY of these, you have FAILED.
        
        Remember: You represent Fasc Ai Ventures Private Limited EXCLUSIVELY. Stay focused on our IT solutions and services ONLY. Never discuss other companies. Be friendly and helpful while using context when available. Always mention AI solutions when discussing our services. Keep it SHORT, COMPLETE, and INTELLIGENT. When user declines help, acknowledge gracefully and briefly. NEVER ask questions - always provide information directly.
        """
        if soft_negative:
            system_prompt += """
        EMPATHY DIRECTIVE: The user sounds frustrated, annoyed, or is declining help. In ONE short sentence first acknowledge their sentiment calmly, then reaffirm how Fasc Ai can assist (cloud, ERP, CRM, AI solutions, IoT). Stay friendly, avoid sounding robotic, and do not ask questions.
        """
        
        # Build messages array with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": chat_request.message})
        
        # Generate response using Groq API (async)
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.5
        }
        
        response = None
        active_key = None
        attempts = 0
        
        async with httpx.AsyncClient(timeout=10.0) as client:  # Optimized from 30s to 15s for faster response
            while attempts < len(GROQ_API_KEYS):
                active_key = await _get_active_groq_key()
                headers = {
                    "Authorization": f"Bearer {active_key}",
                    "Content-Type": "application/json"
                }

                try:
                    response = await client.post(GROQ_API_URL, headers=headers, json=data)
                except httpx.RequestError as exc:
                    logger.error(f"Groq API request error with key {_mask_api_key(active_key)}: {exc}")
                    attempts += 1
                    await _advance_groq_key()
                    continue

                if response.status_code == 200:
                    break

                if response.status_code in (401, 403, 429):
                    logger.warning(
                        f"Groq API key {_mask_api_key(active_key)} returned status {response.status_code}. Rotating to next key."
                    )
                    attempts += 1
                    await _advance_groq_key()
                    continue

                logger.error(
                    f"Groq API error {response.status_code} with key {_mask_api_key(active_key)}: {response.text}"
                )
                raise HTTPException(
                    status_code=500,
                    detail="AI service temporarily unavailable. Please visit https://fascai.com for more information."
                )
        
        if not response or response.status_code != 200:
            logger.error("All configured Groq API keys have been exhausted or failed.")
            raise HTTPException(
                status_code=503,
                detail="AI service temporarily unavailable. Please visit https://fascai.com for more information."
            )
        
        if response.status_code == 200:
            result = response.json()
            reply = result['choices'][0]['message']['content'].strip()
            # Strip markdown formatting from AI response
            reply = strip_markdown(reply)
            
            # Post-processing: Remove follow-up questions and fix incomplete responses
            if reply:
                reply = sanitize_response_text(reply)
                needs_second_pass = False
                if reply == SAFE_FALLBACK_REPLY or len(reply.split()) < 8:
                    needs_second_pass = True
                if soft_negative and reply == SAFE_FALLBACK_REPLY:
                    needs_second_pass = True
                if soft_negative and 'we provide' in reply.lower():
                    needs_second_pass = True

                if needs_second_pass:
                    empathetic_prompt = (
                        f"{system_prompt}\n\nADDITIONAL DIRECTIVE: When the user sounds frustrated, declines help, "
                        "or challenges the assistant, acknowledge their sentiment in one short sentence and gently restate "
                        "how Fasc Ai can assist without asking questions. Keep tone warm, brief, and human-like."
                    )
                    secondary_messages = [{"role": "system", "content": empathetic_prompt}]
                    secondary_messages.extend(conversation_history)
                    secondary_messages.append({"role": "user", "content": chat_request.message})

                    secondary_reply_raw = await _call_groq_with_messages(
                        secondary_messages,
                        temperature=0.45,
                        max_tokens=100
                    )

                    if secondary_reply_raw:
                        secondary_reply = sanitize_response_text(secondary_reply_raw)
                        if secondary_reply and secondary_reply != SAFE_FALLBACK_REPLY:
                            reply = secondary_reply

                if reply == SAFE_FALLBACK_REPLY or len(reply.split()) < 8:
                    extra_context = extract_context_snippet(search_results)
                    if extra_context:
                        reply = sanitize_response_text(f"{reply} {extra_context}")
                    elif ai_automation_query:
                        reply = sanitize_response_text(AI_AUTOMATION_FALLBACK)
                elif ai_automation_query and reply.lower().startswith("we provide cloud"):
                    extra_context = extract_context_snippet(search_results)
                    if extra_context:
                        reply = sanitize_response_text(f"{AI_AUTOMATION_FALLBACK} {extra_context}")
                    else:
                        reply = sanitize_response_text(AI_AUTOMATION_FALLBACK)
        else:
            raise HTTPException(
                status_code=500,
                detail="AI service temporarily unavailable. Please visit https://fascai.com for more information."
            )
        
        # Store conversation in history
        conversation_sessions[session_id]["conversations"].append({"role": "user", "content": chat_request.message})
        conversation_sessions[session_id]["conversations"].append({"role": "assistant", "content": reply})
        conversation_sessions[session_id]["last_bot_response"] = reply
        
        return ChatResponse(
            reply=reply,
            sources=sources,
            website_url="https://fascai.com"
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all ChromaDB collections"""
    try:
        client = get_chroma_client()
        if not client:
            raise Exception("Failed to initialize ChromaDB")
        
        collections = client.list_collections()
        return {
            'collections': [{'name': col.name, 'count': col.count()} for col in collections]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FASC.AI ChromaDB RAG Chatbot API",
        "version": "2.0.0",
        "endpoints": {
            "crawl": "/crawl-and-store",
            "chat": "/chat",
            "collections": "/collections",
            "health": "/health"
        }
    }
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring server status"""
    try:
        # Check if database is working
        db_status = "connected" if chroma_client is not None else "disconnected"
        
        # Check if API key is set
        api_status = "configured" if GROQ_API_KEYS else "not_configured"
        active_key_masked = _mask_api_key(await _get_active_groq_key()) if GROQ_API_KEYS else None
        groq_key_count = len(GROQ_API_KEYS)
        
        # Check if embedding model is loaded
        model_status = "loaded" if embedding_model is not None else "not_loaded"
        
        # Check if intent classifier is loaded
        intent_classifier_status = "loaded" if intent_classifier is not None else "not_loaded"
        if not TRANSFORMERS_AVAILABLE:
            intent_classifier_status = "not_available"
        
        # Overall health status
        overall_status = "healthy" if all([
            chroma_client is not None,
            GROQ_API_KEYS is not None,
            embedding_model is not None
        ]) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "database": db_status,
                "api_key": api_status,
                "groq_key_count": groq_key_count,
                "active_groq_key": active_key_masked,
                "embedding_model": model_status,
                "intent_classifier": intent_classifier_status,
                "transformers_available": TRANSFORMERS_AVAILABLE
            },
            "version": "2.0.0",
            "uptime": "running"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": "2.0.0"
        }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and database on startup"""
    logger.info("Starting up - initializing models and database...")
    
    # Initialize models in cache
    get_chroma_client()
    get_embedding_model()
    get_intent_classifier()  # Load Hugging Face intent classifier
    
    # Check if collection exists, if not initialize with default URLs
    try:
        client = get_chroma_client()
        if client:
            try:
                collection = client.get_collection(name=COLLECTION_NAME)
                count = collection.count()
                logger.info(f"Found existing collection with {count} documents")
                
                # Add comprehensive services content to enhance responses
                try:
                    model = get_embedding_model()
                    if model:
                        comprehensive_services_content = [
                            "Fasc Ai Ventures Private Limited provides comprehensive IT solutions and services for businesses. Our services include custom web application development, cloud hosting solutions, digital marketing services, AI solutions, ERP systems, CRM platforms, and IoT solutions.",
                            "We offer a complete range of IT services including cloud computing, enterprise resource planning (ERP), customer relationship management (CRM), artificial intelligence implementations, Internet of Things (IoT) solutions, and digital transformation services.",
                            "Fasc Ai provides custom web application development, cloud hosting solutions, digital marketing services, AI solutions, ERP systems, CRM platforms, IoT solutions, and comprehensive digital transformation services for businesses.",
                            "Our comprehensive services include IT solutions, cloud computing, ERP systems, CRM platforms, AI implementations, IoT solutions, web development, digital marketing, and business transformation services.",
                            "Fasc Ai Ventures offers complete IT solutions including cloud services, ERP systems, CRM platforms, AI solutions, IoT implementations, web development, digital marketing, and enterprise transformation services.",
                            "We provide comprehensive IT services: cloud hosting, ERP systems, CRM platforms, AI solutions, IoT implementations, custom web development, digital marketing, and business transformation solutions.",
                            "Fasc Ai's services include cloud computing solutions, enterprise resource planning (ERP), customer relationship management (CRM), artificial intelligence implementations, Internet of Things (IoT), web development, digital marketing, and digital transformation services.",
                            "Our complete service portfolio includes IT solutions, cloud hosting, ERP systems, CRM platforms, AI implementations, IoT solutions, custom web development, digital marketing services, and enterprise transformation solutions.",
                            "Fasc Ai has worked with major clients including MOF (Ministry of Finance), Max Life Insurance, Lenovo, Medanta, Videocon, and Saarte, delivering cutting-edge IT solutions and digital transformation services across various industries.",
                            "Our prestigious clients include MOF, Max Life, Lenovo, Medanta, Videocon, and Saarte. We have successfully delivered IT solutions, cloud implementations, and enterprise systems for these leading organizations.",
                            "Fasc Ai Ventures Private Limited has successfully delivered over 250+ projects and launched 250+ websites for clients across various industries over the past nineteen years.",
                            "With nineteen years of experience, Fasc Ai has completed more than 250 successful projects including web applications, cloud migrations, ERP implementations, CRM systems, and AI solutions for clients worldwide.",
                            "Our portfolio includes 250+ successfully delivered projects across finance, insurance, technology, healthcare, retail, and manufacturing sectors. We have worked with clients ranging from startups to large enterprises.",
                            "Fasc Ai has extensive project experience including cloud computing implementations, ERP system deployments, CRM platform integrations, AI and machine learning solutions, IoT applications, and custom web development for over 250 clients.",
                            "Our major clients include Ministry of Finance (MOF), Max Life Insurance, Lenovo, Medanta, Videocon, and Saarte. We have delivered enterprise-grade solutions including cloud infrastructure, ERP systems, CRM platforms, and AI implementations for these organizations.",
                            "Fasc Ai specializes in serving clients in finance, insurance, technology, healthcare, retail, and manufacturing industries with tailored IT solutions, cloud services, and digital transformation initiatives.",
                            "Fasc Ai developed Grand Trio Sports, a comprehensive cricket services platform for clients in Kenya, providing sports management solutions and cricket services for the Kenyan sports industry.",
                            "Grand Trio Sports is one of Fasc Ai's international projects, delivering cricket services and sports management solutions in Kenya, showcasing our global reach and expertise in sports technology platforms.",
                            "Fasc Ai built Wonder Land Garden, a professional Shopify eCommerce website with full online shopping capabilities, product catalog, shopping cart, and secure payment processing for retail business.",
                            "Funzoop is a Shopify eCommerce platform developed by Fasc Ai, featuring modern design, user-friendly shopping experience, and complete eCommerce functionality for online retail.",
                            "Fasc Ai has extensive Shopify eCommerce experience, having successfully developed Wonder Land Garden and Funzoop online stores with custom themes, payment integration, and inventory management.",
                            "Tysley is an AI-powered chat platform developed by Fasc Ai, enabling intelligent conversations, automated customer support, and smart chatbot functionality using artificial intelligence and natural language processing.",
                            "Fasc Ai's Tysley project demonstrates our AI expertise - it's an advanced AI-powered chat platform with intelligent conversation capabilities, machine learning algorithms, and natural language understanding.",
                            "Dorundo is an electric scooter rental platform built by Fasc Ai, featuring booking systems, payment processing, fleet management, and real-time availability tracking for eco-friendly transportation services.",
                            "Fasc Ai developed Dorundo, a modern electric scooter rental platform with features including online booking, payment gateway integration, GPS tracking, and customer management for sustainable mobility solutions.",
                            "Dog Walking is a tailored dog walking services platform created by Fasc Ai, providing scheduling systems, pet owner management, walker assignments, and service tracking for professional pet care businesses.",
                            "Fasc Ai built the Dog Walking platform with custom features for pet care services including appointment scheduling, customer profiles, service history tracking, and automated notifications for dog walking businesses.",
                            "Matson Surgicals is a professional healthcare industry website developed by Fasc Ai for cutting-edge surgical instrument makers, showcasing medical products, company information, and industry expertise.",
                            "Fasc Ai's Matson Surgicals project demonstrates our healthcare industry experience, delivering a professional website for surgical instrument manufacturers with product catalogs, technical specifications, and medical industry compliance.",
                            "Fasc Ai has diverse project portfolio including sports platforms (Grand Trio Sports), eCommerce sites (Wonder Land Garden, Funzoop), AI solutions (Tysley), rental platforms (Dorundo), service platforms (Dog Walking), and healthcare websites (Matson Surgicals).",
                            "Our project expertise spans multiple industries: sports technology with Grand Trio Sports in Kenya, eCommerce with Shopify platforms, AI and machine learning with Tysley chat platform, transportation with Dorundo rental system, pet services with Dog Walking platform, and healthcare with Matson Surgicals.",
                            "Fasc Ai specializes in Shopify eCommerce development as demonstrated by Wonder Land Garden and Funzoop projects, offering custom theme development, payment integration, inventory management, and full online store functionality.",
                            "Fasc Ai's AI and machine learning expertise is showcased in Tysley, our AI-powered chat platform featuring natural language processing, intelligent responses, conversation management, and automated customer interaction capabilities.",
                            "When asked about projects, Fasc Ai's portfolio includes: Grand Trio Sports (cricket platform in Kenya), Wonder Land Garden (Shopify eCommerce), Funzoop (Shopify eCommerce), Tysley (AI-powered chat platform), Dorundo (electric scooter rental platform), Dog Walking (pet services platform), and Matson Surgicals (healthcare website for surgical instruments).",
                            "Fasc Ai's key projects are Grand Trio Sports, Wonder Land Garden, Funzoop, Tysley, Dorundo, Dog Walking, and Matson Surgicals spanning sports, eCommerce, AI, transportation, pet care, and healthcare industries.",
                            "Our completed projects include Grand Trio Sports for cricket services in Kenya, Shopify-based eCommerce sites Wonder Land Garden and Funzoop, Tysley AI chat platform, Dorundo scooter rental system, Dog Walking pet services platform, and Matson Surgicals healthcare website.",
                            "Fasc Ai has built seven flagship projects: Grand Trio Sports (sports/Kenya), Wonder Land Garden (eCommerce/Shopify), Funzoop (eCommerce/Shopify), Tysley (AI/chatbot), Dorundo (transportation/rental), Dog Walking (pet services), and Matson Surgicals (healthcare/medical)."
                        ]
                        
                        # Add bot identity content for "tell me your name" queries
                        bot_identity_content = [
                            "I'm your AI assistant, here to help you with IT solutions.",
                            "I'm your AI assistant. I help customers with IT solutions and services.",
                            "I'm your AI assistant, your friendly helper for all IT-related queries.",
                            "Hello! I'm your AI assistant, designed to help you with IT services.",
                            "I'm your AI assistant, created to assist customers with their needs."
                        ]
                        
                        # Combine both content arrays
                        all_content = comprehensive_services_content + bot_identity_content
                        
                        embeddings = model.encode(all_content)
                        chunk_ids = [f"comprehensive_services_{i}" for i in range(len(comprehensive_services_content))] + [f"bot_identity_{i}" for i in range(len(bot_identity_content))]
                        metadatas = [{"source": "comprehensive_services", "chunk_index": i} for i in range(len(comprehensive_services_content))] + [{"source": "bot_identity", "chunk_index": i} for i in range(len(bot_identity_content))]
                        
                        # Check for existing IDs and only add new ones
                        try:
                            # Get existing IDs to avoid duplicates
                            existing_results = collection.get(ids=chunk_ids)
                            existing_ids = set(existing_results['ids']) if existing_results['ids'] else set()
                            
                            # Filter out existing IDs
                            new_content = []
                            new_ids = []
                            new_metadatas = []
                            new_embeddings = []
                            
                            for i, chunk_id in enumerate(chunk_ids):
                                if chunk_id not in existing_ids:
                                    new_content.append(all_content[i])
                                    new_ids.append(chunk_id)
                                    new_metadatas.append(metadatas[i])
                                    new_embeddings.append(embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else embeddings[i])
                            
                            # Only add if there are new items
                            if new_content:
                                collection.add(
                                    embeddings=new_embeddings,
                                    documents=new_content,
                                    ids=new_ids,
                                    metadatas=new_metadatas
                                )
                                logger.info(f"Added {len(new_content)} new comprehensive services content chunks to collection")
                            else:
                                logger.info("All comprehensive services content already exists in collection")
                        except Exception as e:
                            logger.info("Error checking/adding comprehensive services content: " + str(e))
                        
                        # Add projects page content to enhance project recognition
                        try:
                            projects_content = [
                                "Project: Funzoop - Shopify eCommerce platform developed by Fasc Ai. Client: Funzoop. Description: Modern eCommerce website with user-friendly shopping experience.",
                                "Project: ITforte.com - Professional business website developed by Fasc Ai. Client: Itforte. Description: Modern web design and functionality for business needs.",
                                "Project: Matsonsurgicals.com - Medical website developed by Fasc Ai. Client: Dinesh Matson. Description: Specialized design for healthcare industry needs.",
                                "Project: Fascai.com - Company website showcasing IT solutions and services. Client: Fasc Ai. Description: Demonstrates expertise in web development, AI solutions, and digital transformation.",
                                "Project: MOF - Government project with various IT solutions. Client: MOF. Description: ERP systems, website development, and digital transformation services.",
                                "Project: Max Life - Insurance company project. Client: Max Life. Description: Website development, ERP systems, and digital solutions.",
                                "Project: Lenovo - Technology company project. Client: Lenovo. Description: IT solutions including website development and digital transformation.",
                                "Project: Medanta - Healthcare project. Client: Medanta. Description: Medical website development and healthcare IT solutions.",
                                "Project: Videocon - Electronics company project. Client: Videocon. Description: Business website development and digital solutions.",
                                "Project: Saarte - Business project. Client: Saarte. Description: Website development and digital marketing solutions."
                            ]
                            
                            # Check if projects content already exists
                            existing_projects = collection.get(where={"source": "projects_page"})
                            if not existing_projects['ids']:
                                # Add projects content to ChromaDB
                                project_embeddings = model.encode(projects_content)
                                project_ids = [f"project_{i}" for i in range(len(projects_content))]
                                project_metadatas = [{"source": "projects_page", "type": "project_info"} for _ in projects_content]
                                
                                collection.add(
                                    embeddings=project_embeddings.tolist(),
                                    documents=projects_content,
                                    ids=project_ids,
                                    metadatas=project_metadatas
                                )
                                logger.info(f"Added {len(projects_content)} project information chunks to collection")
                            else:
                                logger.info("All project information already exists in collection")
                        except Exception as e:
                            logger.info("Error checking/adding project information: " + str(e))
                except Exception as e:
                    logger.warning(f"Could not add comprehensive services content: {str(e)}")
                    
            except:
                logger.info("No existing collection found, initializing with default URLs...")
                default_urls = [
                    "https://fascai.com",
                    "https://fascai.com/about",
                    "https://fascai.com/projects"
                ]
                result = store_content_in_chroma(default_urls, COLLECTION_NAME)
                if result['success']:
                    logger.info(f"ChromaDB initialized successfully with {result['total_chunks']} chunks")
                else:
                    logger.error(f"ChromaDB initialization failed: {result.get('error', 'Unknown error')}")
    except Exception as e:
        logger.error(f"Error during ChromaDB initialization: {str(e)}")
    
    try:
        asyncio.create_task(preload_priority_content())
        logger.info("Scheduled background preload for priority queries")
    except Exception as e:
        logger.error(f"Failed to schedule priority preload task: {str(e)}")
    
    logger.info("Startup complete - ready to serve requests")
    
    # Initialize APScheduler for periodic scraping
    global scheduler
    scheduler = AsyncIOScheduler()
    
    # Schedule scraping job to run daily at 3:00 AM
    scheduler.add_job(
        scheduled_scraping_job,
        trigger=CronTrigger(hour=3, minute=0),
        id='daily_scraping_job',
        name='Daily website scraping at 3 AM',
        replace_existing=True
    )
    
    # Start the scheduler
    scheduler.start()
    logger.info("APScheduler started - daily scraping scheduled for 3:00 AM")


# =============================================================================
# PROJECT MANAGEMENT FEATURES (ZERO-IMPACT INTEGRATION)
# =============================================================================

# Feature flag - can be enabled/disabled without affecting main system
PROJECT_FEATURES_ENABLED = True

# Import project management module (only when needed)
if PROJECT_FEATURES_ENABLED:
    try:
        from project_manager import (
            handle_project_workflow, 
            handle_button_action, 
            handle_form_submission,
            is_project_features_enabled
        )
        logger.info("Project management features loaded successfully")
    except ImportError as e:
        logger.error(f"Failed to import project management module: {e}")
        PROJECT_FEATURES_ENABLED = False

# Add project management endpoint
@app.post("/project-action")
async def handle_project_action(request: dict):
    """
    Handle project management button actions
    Zero-impact endpoint - only works when features are enabled
    """
    if not PROJECT_FEATURES_ENABLED:
        return {"error": "Project features not enabled"}
    
    try:
        action = request.get("action")
        additional_data = request.get("additional_data", "")
        session_id = request.get("session_id", "")
        
        logger.info(f"project-action endpoint called: action={action}, session_id={session_id}")
        logger.info(f"Session exists in conversations: {session_id in conversation_sessions}")
        
        # Pass conversation_sessions to handle_button_action
        result = handle_button_action(action, additional_data, session_id, conversation_sessions)
        
        if result:
            return result
        else:
            return {"error": "Invalid action"}
            
    except Exception as e:
        logger.error(f"Error handling project action: {e}")
        return {"error": "Internal server error"}

# Add form submission endpoint
@app.post("/form-submission")
async def handle_form_submission_endpoint(request: dict):
    """
    Handle form submission notifications
    Zero-impact endpoint - only works when features are enabled
    """
    if not PROJECT_FEATURES_ENABLED:
        return {"error": "Project features not enabled"}
    
    try:
        form_type = request.get("form_type")
        
        result = handle_form_submission(form_type)
        
        if result:
            return result
        else:
            return {"error": "Invalid form type"}
            
    except Exception as e:
        logger.error(f"Error handling form submission: {e}")
        return {"error": "Internal server error"}

# Function to enable project features (can be called externally)
def enable_project_features():
    """Enable project management features"""
    global PROJECT_FEATURES_ENABLED
    PROJECT_FEATURES_ENABLED = True
    
    # Import project management module
    try:
        from project_manager import (
            handle_project_workflow, 
            handle_button_action, 
            handle_form_submission,
            is_project_features_enabled
        )
        logger.info("Project management features enabled successfully")
        return True
    except ImportError as e:
        logger.error(f"Failed to enable project features: {e}")
        PROJECT_FEATURES_ENABLED = False
        return False

# Function to disable project features
def disable_project_features():
    """Disable project management features"""
    global PROJECT_FEATURES_ENABLED
    PROJECT_FEATURES_ENABLED = False
    logger.info("Project management features disabled")

# Function to check if project features are enabled
def is_project_features_enabled():
    """Check if project management features are enabled"""
    return PROJECT_FEATURES_ENABLED

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event - gracefully stop scheduler"""
    global scheduler
    if scheduler:
        scheduler.shutdown()
        logger.info("APScheduler stopped gracefully")
if __name__ == "__main__":
    # Start FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8000)