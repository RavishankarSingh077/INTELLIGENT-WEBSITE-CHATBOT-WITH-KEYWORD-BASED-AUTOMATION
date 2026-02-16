#!/usr/bin/env python3
"""
Project Management Module for Fasc AI Chatbot
Zero-impact integration with existing chatbot system
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# Feature flag - can be enabled/disabled without affecting main system
PROJECT_FEATURES_ENABLED = False  # Default: OFF for safety

# Form URLs mapping
FORM_URLS = {
    'new_project': 'https://portal.fascai.com/register.php',
    'existing_project': 'https://portal.fascai.com/index.php?rp=/login',
    'add_features': 'https://portal.fascai.com/index.php?rp=/login',
    'raise_ticket': 'https://portal.fascai.com/index.php?rp=/login',
    'complaint': 'https://portal.fascai.com/register.php'
}

def is_project_intent(message: str) -> bool:
    """
    Check if message indicates user wants to work on a project
    Returns True if project intent detected, False otherwise
    """
    message_lower = message.lower().strip()
    
    # Project intent patterns
    project_patterns = [
        'want to start a project', 'want to work with you', 'start a project with you',
        'work on a project', 'begin a project', 'new project', 'start project',
        'collaborate with you', 'partner with you', 'hire you', 'engage with you',
        'get started', 'begin working', 'start working together',
        'project with fasc ai', 'project with fascai', 'work with fasc ai',
        'interested in your services', 'interested in working', 'want your services',
        'i want to start a project', 'i want to work with you', 'want to start',
        'i want to work', 'want to work', 'start a project', 'want to begin',
        'i need a project', 'need a project', 'looking for project', 'project help',
        'work together', 'collaborate', 'partnership', 'project partnership',
        # Service request patterns
        'i need a website', 'i want a website', 'need a website', 'want a website',
        'i need a chatbot', 'i want a chatbot', 'need a chatbot', 'want a chatbot',
        'i need erp', 'i want erp', 'need erp', 'want erp',
        'i need crm', 'i want crm', 'need crm', 'want crm',
        'i need cloud', 'i want cloud', 'need cloud', 'want cloud',
        'i need ai', 'i want ai', 'need ai', 'want ai',
        'i need iot', 'i want iot', 'need iot', 'want iot',
        'website development', 'chatbot development', 'erp implementation',
        'crm implementation', 'cloud services', 'ai solutions', 'iot solutions',
        # Contact/team connection patterns
        'how can i connect', 'how to connect', 'connect with you', 'connect with team',
        'contact you', 'contact team', 'contact your team', 'reach you', 'reach team',
        'get in touch', 'get in contact', 'speak with', 'talk with', 'meet with',
        'where is form', 'contact form', 'project form', 'get started', 'start working',
        'work with fasc', 'work with fasc ai', 'fasc ai team', 'your team',
        'team contact', 'team connection', 'team communication', 'team collaboration'
    ]
    
    # Check for project intent patterns
    for pattern in project_patterns:
        if pattern in message_lower:
            return True
    
    # Check for combination of keywords
    project_keywords = ['project', 'work', 'start', 'begin', 'collaborate', 'partner', 'hire']
    positive_keywords = ['want', 'interested', 'would like', 'need', 'looking for']
    
    has_project_keyword = any(keyword in message_lower for keyword in project_keywords)
    has_positive_keyword = any(keyword in message_lower for keyword in positive_keywords)
    
    if has_project_keyword and has_positive_keyword:
        return True
    
    return False

def is_existing_customer_query(message: str) -> bool:
    """
    Check if user is indicating they are an existing customer
    """
    message_lower = message.lower().strip()
    
    # Patterns that indicate existing customer status
    existing_customer_patterns = [
        'i am already your customer', 'i am your existing customer', 
        'i am your customer', 'i am your client', 'i am already your client',
        'i\'m already your customer', 'i\'m your existing customer',
        'i\'m your customer', 'i\'m your client', 'i\'m already your client',
        'we are existing customers', 'we are already customers',
        'i have worked with you before', 'we have worked with you',
        'i am already a customer', 'i\'m already a customer',
        'we are your customers', 'we are already your customers',
        'existing customer', 'already customer', 'current customer',
        'i am your existing client', 'i\'m your existing client',
        # Additional patterns for better detection
        'i am fasc ai client', 'i am fascai client', 'i am fasc client',
        'i\'m fasc ai client', 'i\'m fascai client', 'i\'m fasc client',
        'i made my software through you', 'i made software through you',
        'made my software with you', 'made software with you',
        'built my software with you', 'built software with you',
        'developed my software with you', 'developed software with you',
        'created my software with you', 'created software with you',
        'i am fasc ai customer', 'i am fascai customer', 'i am fasc customer',
        'i\'m fasc ai customer', 'i\'m fascai customer', 'i\'m fasc customer',
        'we are fasc ai customers', 'we are fascai customers', 'we are fasc customers',
        'fasc ai client', 'fascai client', 'fasc client',
        'fasc ai customer', 'fascai customer', 'fasc customer'
    ]
    
    # Check if message contains existing customer indicators
    if any(pattern in message_lower for pattern in existing_customer_patterns):
        return True
    
    return False

def is_projects_inquiry(message: str) -> bool:
    """
    Check if user is asking about projects in general (not providing project details)
    """
    message_lower = message.lower().strip()
    
    # Patterns that indicate general project inquiries
    projects_inquiry_patterns = [
        'tell me about all your projects', 'show me your projects', 'what projects have you done',
        'list your projects', 'show me projects', 'tell me about projects',
        'what projects do you have', 'your projects', 'about your projects',
        'projects you have done', 'projects you worked on', 'all your projects',
        'tell me about the projects', 'show me all projects', 'what are your projects',
        'projects list', 'your project list', 'tell me projects', 'show projects',
        'what projects', 'tell me about projects you have done'
    ]
    
    # Check for projects inquiry patterns
    if any(pattern in message_lower for pattern in projects_inquiry_patterns):
        return True
    
    # Check for question patterns about projects
    if (any(word in message_lower for word in ['tell', 'show', 'list', 'what']) and
        any(word in message_lower for word in ['projects', 'project']) and
        any(word in message_lower for word in ['your', 'you', 'all'])):
        return True
    
    return False

def is_single_project_showcase(message: str) -> bool:
    """
    Check if user is asking about a specific project by name (for showcase purposes)
    """
    message_lower = message.lower().strip()
    
    # REAL CUSTOMER WHITELIST - Only these are actual customers from website
    known_project_names = [
        'funzoop', 'itforte', 'itforte.com', 'matsonsurgicals', 'matsonsurgicals.com',
        'mof', 'max life', 'lenovo', 'medanta', 'videocon',
        'dorundo', 'grand trio', 'wonder land', 'tysley', 'dog walking', 'matson surgicals',
        'saarte', 'grand trio sports', 'wonder land garden'
    ]
    
    # BLOCKED DOMAINS - These should NEVER be treated as customers
    blocked_domains = [
        'fascai.com', 'google.com', 'facebook.com', 'twitter.com', 'instagram.com',
        'linkedin.com', 'youtube.com', 'github.com', 'stackoverflow.com',
        'amazon.com', 'microsoft.com', 'apple.com', 'netflix.com', 'spotify.com',
        'wikipedia.org', 'reddit.com', 'pinterest.com', 'tiktok.com', 'snapchat.com'
    ]
    
    # Check if it's a blocked domain first
    for blocked_domain in blocked_domains:
        if blocked_domain in message_lower:
            return False
    
    # Check if it's a single word/simple phrase that matches known projects
    if len(message.split()) <= 2:  # 1-2 words max
        # Check for exact matches or similar names
        for project_name in known_project_names:
            if (project_name in message_lower or 
                message_lower in project_name or
                message_lower.replace('.', '') == project_name.replace('.', '')):
                return True
        
        # Check for domain-like patterns (contains dots)
        if '.' in message and len(message.split()) == 1:
            return True
    
    return False

def validate_company_in_chromadb(company_name: str, search_function=None) -> bool:
    """
    Validate if a company exists as a real client in ChromaDB
    Returns True only if the company is found in the actual project database
    """
    try:
        logger.info(f"validate_company_in_chromadb called for: {company_name}")
        
        if not search_function:
            logger.info("No search function provided, trying to get from app_chromadb")
            # If no search function provided, try to get it from app_chromadb
            try:
                import app_chromadb
                search_function = app_chromadb.search_collection
                logger.info("Successfully got search function from app_chromadb")
            except Exception as e:
                logger.warning(f"Could not get search function: {e}")
                return False
        else:
            logger.info("Using provided search function")
        
        # Create search queries to find the company in ChromaDB
        queries = [
            f"project {company_name}",
            f"client {company_name}",
            f"{company_name} project",
            f"{company_name} website",
            f"{company_name} development",
            f"website {company_name}",
            f"project details {company_name}"
        ]
        
        found_results = []
        
        for query in queries:
            try:
                logger.info(f"Searching ChromaDB with query: {query}")
                results = search_function(query)
                if results and len(results) > 0:
                    logger.info(f"Found {len(results)} results for query: {query}")
                    found_results.extend(results)
                else:
                    logger.info(f"No results for query: {query}")
            except Exception as e:
                logger.warning(f"Error searching ChromaDB with query '{query}': {e}")
                continue
        
        # Check if any results contain the company name
        for result in found_results:
            content = result.get('content', '').lower()
            if company_name.lower() in content:
                logger.info(f"Company {company_name} found in ChromaDB: {content[:100]}...")
                return True
        
        logger.info(f"Company {company_name} not found in ChromaDB")
        return False
        
    except Exception as e:
        logger.error(f"Error validating company in ChromaDB: {e}")
        return False

def is_existing_project_query(message: str, session_context: dict = None, search_function=None) -> bool:
    """
    Check if user is providing project details for existing project lookup
    DATABASE-FIRST validation: Only treat companies as customers if they exist in ChromaDB
    """
    message_lower = message.lower().strip()
    logger.info(f"is_existing_project_query called with message: {message}, session_context: {session_context}")
    
    # If user clicked "Existing Project" button, check whitelist FIRST (before any other checks)
    # This ensures 100% accuracy for known customers and prevents false negatives
    if session_context and session_context.get('waiting_for_existing_project', False):
        logger.info(f"User is in existing project context for message: {message}")
        logger.info(f"Session context: {session_context}")
        
        # REAL CUSTOMER WHITELIST - Only these are actual customers from website
        known_project_names = [
            'funzoop', 'itforte', 'itforte.com', 'matsonsurgicals', 'matsonsurgicals.com',
            'mof', 'max life', 'lenovo', 'medanta', 'videocon',
            'dorundo', 'grand trio', 'wonder land', 'tysley', 'dog walking', 'matson surgicals',
            'saarte', 'grand trio sports', 'wonder land garden'
        ]
        
        # STEP 1: WHITELIST CHECK FIRST (before blocked domains and exclusion patterns)
        # This ensures 100% accuracy for known customers (1-2 word messages)
        if len(message.split()) <= 2:  # 1 or 2 words
            message_lower = message.lower().strip()
            
            # Check whitelist directly (handles both single and multi-word names)
            for project_name in known_project_names:
                if (project_name in message_lower or 
                    message_lower in project_name or
                    message_lower.replace('.', '') == project_name.replace('.', '')):
                    logger.info(f"Detected existing project query: {message} (whitelist match)")
                    return True
            
            # If not in whitelist, return False (don't treat as existing project)
            logger.info(f"Message '{message}' not in whitelist - not treating as existing project")
            return False
    
    # ENHANCED BLOCKED DOMAINS - These should NEVER be treated as customers (CHECK FIRST!)
    blocked_domains = [
        # Major tech companies (with and without .com)
        'google', 'facebook', 'twitter', 'instagram', 'linkedin', 'youtube', 'github', 
        'stackoverflow', 'amazon', 'microsoft', 'apple', 'netflix', 'spotify',
        'wikipedia', 'reddit', 'pinterest', 'tiktok', 'snapchat', 'meta',
        # Indian tech companies
        'flipkart', 'myntra', 'paytm', 'zomato', 'swiggy', 'uber', 'ola', 
        'make my trip', 'booking', 'ola cabs', 'swiggy', 'zomato',
        # Our own domain
        'fascai.com', 'fascai', 'fasc ai'
    ]
    
    # Check if it's a blocked domain first - REJECT IMMEDIATELY
    for blocked_domain in blocked_domains:
        if blocked_domain in message_lower:
            logger.info(f"Blocked domain detected: {blocked_domain} in message: {message}")
            return False
    
    # Additional check for exact matches (case insensitive)
    if message_lower in [domain.lower() for domain in blocked_domains]:
        logger.info(f"Exact blocked domain match: {message_lower}")
        return False
    
    # EXCLUSION PATTERNS - These should NEVER be treated as project queries
    exclusion_patterns = [
        # Common phrases
        'great', 'thank you', 'thanks', 'ok', 'okay', 'yes', 'no', 'sure',
        'good', 'bad', 'fine', 'alright', 'perfect', 'excellent', 'wonderful',
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'what\'s up', 'whats up', 'how do you do',
        
        # Personal introductions
        'i am', 'i\'m', 'my name is', 'i\'m called', 'people call me',
        'abhishek', 'ravi', 'john', 'smith', 'kumar', 'jha', 'singh',
        
        # Questions and requests
        'how can you', 'how do you', 'can you', 'do you', 'will you',
        'how to', 'what is', 'what are', 'why', 'when', 'where',
        'tell me', 'show me', 'list', 'about', 'all', 'help',
        'i need help', 'i want help', 'can you help', 'please help',
        
        # Emotional expressions
        'frustrated', 'frustated', 'angry', 'sad', 'happy', 'excited',
        'worried', 'disappointed', 'pleased', 'upset', 'annoyed',
        
        # General conversation
        'i am coming', 'first time', 'new here', 'just started',
        'don\'t think', 'could help', 'connect with', 'team',
        'contact details', 'form', 'website', 'find any',
        
        # Additional emotional and context patterns
        'upset with', 'frustrated with', 'happy with', 'sad about', 'excited about',
        'worried about', 'disappointed with', 'pleased with', 'angry about',
        'annoyed with', 'excited for', 'worried for', 'disappointed in',
        'facing difficulties', 'perform well', 'system aren\'t working', 'can\'t find',
        'need help', 'want help', 'looking for', 'searching for', 'asking about',
        'telling you', 'wanting to', 'needing to', 'trying to', 'going to',
        'i am perform', 'i am facing', 'i am upset', 'i am happy', 'i am sad',
        'i am frustrated', 'i am excited', 'i am worried', 'i am disappointed',
        'i am pleased', 'i am annoyed', 'i am angry', 'i am good', 'i am bad',
        'i am fine', 'i am alright', 'i am perfect', 'i am excellent', 'i am wonderful',
        'i am great', 'i am terrible', 'i am awful', 'i am amazing', 'i am fantastic'
    ]
    
    # Check for exclusion patterns
    for pattern in exclusion_patterns:
        if pattern in message_lower:
            return False
    
    # First check if it's a projects inquiry (should be handled separately)
    if is_projects_inquiry(message):
        return False
    
    # Patterns that indicate project details
    project_detail_patterns = [
        'project name', 'client name', 'my project', 'our project',
        'project is', 'client is', 'working on', 'current project'
    ]
    
    # Check if message contains project-like information
    if any(pattern in message_lower for pattern in project_detail_patterns):
        return True
    
    # Check for comma-separated values (likely project name, client name)
    if ',' in message and len(message.split(',')) == 2:
        parts = message.split(',')
        if len(parts[0].strip()) > 3 and len(parts[1].strip()) > 3:
            return True
    
    # Check for single project names/URLs (VERY STRICT VALIDATION)
    # Only accept if it's a single word, not a full sentence
    if len(message.split()) == 1:  # Only 1 word
        # Look for domain-like patterns (contains dots)
        if '.' in message:
            # Single word with dots - likely a domain/URL
            return True
        
        # Check for project names that look like websites/domains
        if (message.count('.') >= 1 and 
            not any(char in message for char in [' ', ',', '?', '!', ':', ';']) and
            len(message) > 3):
            return True
    
    return False

def is_in_known_customers(project_name: str) -> bool:
    """
    Check if project name is in the known customers whitelist
    Returns True if project is in whitelist, False otherwise
    """
    if not project_name:
        return False
    
    # REAL CUSTOMER WHITELIST - Only these are actual customers from website
    known_project_names = [
        'funzoop', 'itforte', 'itforte.com', 'matsonsurgicals', 'matsonsurgicals.com',
        'mof', 'max life', 'lenovo', 'medanta', 'videocon',
        'dorundo', 'grand trio', 'wonder land', 'tysley', 'dog walking', 'matson surgicals',
        'saarte', 'grand trio sports', 'wonder land garden'
    ]
    
    message_lower = project_name.lower().strip()
    
    # Check for exact matches or similar names
    for known_name in known_project_names:
        if (known_name in message_lower or 
            message_lower in known_name or
            message_lower.replace('.', '') == known_name.replace('.', '')):
            return True
    
    return False

def extract_project_details(message: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract project name and client name from message
    Returns (project_name, client_name) or (None, None) if not found
    """
    message = message.strip()
    
    # Handle comma-separated format: "ITforte.com, Dinesh Batra"
    if ',' in message:
        parts = [part.strip() for part in message.split(',')]
        if len(parts) == 2:
            project_name = parts[0].strip()
            client_name = parts[1].strip()
            
            # Validate extracted names
            if is_valid_project_name(project_name) and is_valid_client_name(client_name):
                return project_name, client_name
    
    # Handle single project name/URL: "ITforte.com" or "funzoop"
    if (len(message.split()) == 1 and 
        not any(char in message for char in [',', '?', '!', ':', ';']) and
        is_valid_project_name(message)):
        
        project_name = message.strip()
        # For single project names, we'll extract client name from website scraping
        # For now, use a generic client name or extract from project name
        client_name = extract_client_name_from_project(project_name)
        return project_name, client_name
    
    # Handle other formats
    words = message.split()
    if len(words) >= 2:
        # Try to identify project name (usually first part)
        potential_project = words[0]
        potential_client = ' '.join(words[1:])
        
        if is_valid_project_name(potential_project) and is_valid_client_name(potential_client):
            return potential_project, potential_client
    
    return None, None

def is_valid_project_name(name: str) -> bool:
    """Check if project name is valid (supports both names and URLs)"""
    if not name or len(name) < 2:
        return False
    
    # Remove common prefixes/suffixes
    name = name.strip()
    
    # Check if it looks like a URL/domain
    if '.' in name:
        return True
    
    # Check if it's a reasonable project name (at least 3 characters)
    if len(name) >= 3:
        return True
    
    # Check if it contains alphanumeric characters
    if any(c.isalnum() for c in name):
        return True
    
    return False

def is_valid_client_name(name: str) -> bool:
    """Check if client name is valid"""
    if not name or len(name) < 2:
        return False
    
    name = name.strip()
    
    # Should contain at least one alphabetic character
    if not any(c.isalpha() for c in name):
        return False
    
    return True

def clean_project_name(name: str) -> str:
    """Clean project/client name by removing unnecessary words"""
    if not name:
        return name
    
    # Remove common filler words
    filler_words = ["it's", "sorry", "the", "a", "an", "is", "are", "was", "were"]
    words = name.lower().split()
    cleaned_words = [word for word in words if word not in filler_words]
    
    if cleaned_words:
        return ' '.join(cleaned_words).title()
    return name.title()

def extract_client_name_from_project(project_name: str) -> str:
    """
    Extract or generate a client name from project name
    For single project names like "ITforte.com", try to extract meaningful client name
    """
    # Remove common domain extensions
    name_without_ext = project_name.replace('.com', '').replace('.org', '').replace('.net', '').replace('.in', '')
    
    # If it looks like a company name, use it as client name
    if len(name_without_ext) > 2 and not name_without_ext.isdigit():
        # Capitalize properly
        client_name = name_without_ext.title()
        return client_name
    
    # Fallback: use the project name itself
    return project_name.title()

def search_project_in_chromadb(project_name: str, client_name: str, search_function) -> Dict:
    """
    Search for existing project in ChromaDB using semantic search
    Optimized: Parallel execution of all queries for 10x speed improvement
    """
    try:
        # Create search queries - enhanced for both project names and URLs
        queries = [
            f"project {project_name} client {client_name}",
            f"{project_name} {client_name} project",
            f"client {client_name} project {project_name}",
            f"{project_name} website project",
            f"{client_name} project details",
            f"{project_name} website development",
            f"website {project_name} client {client_name}",
            f"{project_name} project client {client_name}",
            f"{project_name} development project",
            f"project details {project_name} {client_name}"
        ]
        
        found_projects = []
        
        # Parallel execution of all queries - 10x faster than sequential
        def search_single_query(query):
            try:
                results = search_function(query)
                if results and len(results) > 0:
                    return results
            except Exception as e:
                logger.warning(f"Error searching ChromaDB with query '{query}': {e}")
            return []
        
        # Execute all queries in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_query = {executor.submit(search_single_query, query): query for query in queries}
            for future in as_completed(future_to_query):
                try:
                    results = future.result()
                    if results:
                        found_projects.extend(results)
                except Exception as e:
                    query = future_to_query[future]
                    logger.warning(f"Error processing query '{query}': {e}")
                    continue
        
        # Process results to find project matches
        project_matches = []
        
        for result in found_projects:
            content = result.get('content', '').lower()
            if (project_name.lower() in content or 
                client_name.lower() in content or
                any(word in content for word in project_name.lower().split()) or
                any(word in content for word in client_name.lower().split())):
                
                project_matches.append({
                    'content': result.get('content', ''),
                    'metadata': result.get('metadata', {}),
                    'similarity': result.get('similarity', 0)
                })
        
        # If we have any results from ChromaDB searches, consider it found
        # This fixes the issue where ChromaDB finds data but function returns not found
        if project_matches:
            # Return the best match
            best_match = max(project_matches, key=lambda x: x.get('similarity', 0))
            return {
                'found': True,
                'project_details': best_match['content'],
                'metadata': best_match['metadata'],
                'similarity': best_match['similarity']
            }
        
        # If no matches found but we have any results, use the first one
        # This handles cases where ChromaDB finds relevant data but matching logic is too strict
        if found_projects:
            first_result = found_projects[0]
            return {
                'found': True,
                'project_details': first_result.get('content', ''),
                'metadata': first_result.get('metadata', {}),
                'similarity': first_result.get('similarity', 0)
            }
        
        return {'found': False, 'reason': 'No matching project found'}
        
    except Exception as e:
        logger.error(f"Error searching for project in ChromaDB: {e}")
        return {'found': False, 'reason': f'Search error: {str(e)}'}

def generate_project_intent_response(message: str = "") -> str:
    """Generate response for project intent based on specific service request"""
    import random
    
    message_lower = message.lower().strip()
    
    # Chatbot service responses (check first to avoid website match)
    if any(word in message_lower for word in ['chatbot', 'bot']):
        chatbot_responses = [
            "Fantastic! We develop intelligent chatbots just like this one. We can create custom chatbots for your website with natural language processing and AI capabilities.",
            "Great! We're experts in chatbot development and have implemented chatbots for several clients. Let's discuss your chatbot requirements.",
            "Perfect! We can build a custom chatbot for your business with advanced AI capabilities and seamless integration."
        ]
        return random.choice(chatbot_responses)
    
    # Website service responses
    elif any(word in message_lower for word in ['website', 'web development']):
        website_responses = [
            "Perfect! We specialize in custom website development. We can create responsive websites, e-commerce platforms, and web applications. Let's discuss your requirements.",
            "Great choice! We're experts in website development and have created websites for clients like MOF, Max Life, and Lenovo. What type of website do you need?",
            "Excellent! We can help you build a professional website. We offer responsive design, e-commerce solutions, and custom web applications."
        ]
        return random.choice(website_responses)
    
    # ERP service responses
    elif any(word in message_lower for word in ['erp', 'enterprise resource planning']):
        erp_responses = [
            "Excellent! We provide comprehensive ERP solutions to streamline your business processes. We've successfully implemented ERP systems for various industries.",
            "Great choice! Our ERP solutions help businesses manage their operations efficiently. Let's discuss your specific requirements.",
            "Perfect! We can help you implement a robust ERP system tailored to your business needs."
        ]
        return random.choice(erp_responses)
    
    # CRM service responses
    elif any(word in message_lower for word in ['crm', 'customer relationship management']):
        crm_responses = [
            "Fantastic! We develop custom CRM solutions to help you manage customer relationships effectively. We've created CRM systems for various businesses.",
            "Great! Our CRM solutions help businesses track and manage customer interactions. Let's discuss your CRM requirements.",
            "Excellent! We can build a comprehensive CRM system to enhance your customer relationships and sales processes."
        ]
        return random.choice(crm_responses)
    
    # Cloud service responses
    elif any(word in message_lower for word in ['cloud', 'cloud computing']):
        cloud_responses = [
            "Perfect! We provide comprehensive cloud computing solutions including migration, hosting, and cloud-based applications.",
            "Great choice! Our cloud services help businesses scale and optimize their operations. Let's discuss your cloud requirements.",
            "Excellent! We can help you leverage cloud technology for better performance and cost efficiency."
        ]
        return random.choice(cloud_responses)
    
    # AI service responses
    elif any(word in message_lower for word in ['ai', 'artificial intelligence']):
        ai_responses = [
            "Fantastic! We develop AI solutions including machine learning, natural language processing, and intelligent automation.",
            "Great! Our AI implementations help businesses automate processes and gain insights from data. Let's discuss your AI requirements.",
            "Perfect! We can create custom AI solutions to enhance your business operations and decision-making."
        ]
        return random.choice(ai_responses)
    
    # IoT service responses
    elif any(word in message_lower for word in ['iot', 'internet of things']):
        iot_responses = [
            "Excellent! We develop IoT solutions to connect and manage devices for smart operations and data collection.",
            "Great choice! Our IoT implementations help businesses create connected systems and gather valuable insights. Let's discuss your IoT requirements.",
            "Perfect! We can build comprehensive IoT solutions to optimize your business operations and connectivity."
        ]
        return random.choice(iot_responses)
    
    # Generic project responses
    else:
        generic_responses = [
        "We'd be happy to work with you! I can help you get started with your project. What type of project are you looking for?",
        "Excellent! We're excited about the opportunity to work together. Let me help you with your project requirements.",
        "Great! We'd love to collaborate with you on your project. How can I assist you in getting started?",
        "Fantastic! We're here to help you with your project needs. What kind of project are you planning?"
    ]
        return random.choice(generic_responses)

def handle_projects_inquiry() -> Dict:
    """Handle general inquiries about projects and provide project showcase"""
    import random
    
    responses = [
        "Here are some of our successful projects: Funzoop (eCommerce platform), ITforte.com (business website), Matsonsurgicals.com (medical website), and many more. We've worked with clients like MOF, Max Life, Lenovo, Medanta, and Videocon.",
        "We've completed numerous projects including eCommerce platforms, business websites, ERP systems, and AI solutions. Our clients include MOF, Max Life, Lenovo, Medanta, Videocon, and many others.",
        "Our project portfolio includes Funzoop (Shopify eCommerce), ITforte.com, Matsonsurgicals.com, and various other websites and systems for clients like MOF, Max Life, Lenovo, Medanta, and Videocon."
    ]
    
    response_message = random.choice(responses)
    
    # No buttons needed for general project inquiries
    return create_button_response(response_message, [])

def handle_single_project_showcase(message: str) -> Dict:
    """Handle inquiries about specific projects for showcase purposes"""
    message_lower = message.lower().strip()
    
    # Project-specific showcase responses
    if 'funzoop' in message_lower:
        response = "Funzoop is a Shopify eCommerce platform we developed, featuring modern design and user-friendly shopping experience. It's a great example of our eCommerce solutions. Would you like to know more about our eCommerce development services?"
    elif 'itforte' in message_lower or 'itforte.com' in message_lower:
        response = "ITforte.com is a professional business website we created, showcasing modern web design and functionality. It demonstrates our expertise in business website development. Would you like to discuss similar solutions for your business?"
    elif 'matsonsurgicals' in message_lower or 'matsonsurgicals.com' in message_lower:
        response = "Matsonsurgicals.com is a medical website we developed, featuring specialized design for healthcare industry needs. It's an excellent example of our healthcare website solutions. Would you like to explore our medical website development services?"
    elif 'fascai' in message_lower or 'fascai.com' in message_lower:
        response = "Fascai.com is our company website showcasing our IT solutions and services. It demonstrates our expertise in web development, AI solutions, and digital transformation. Would you like to learn more about our comprehensive IT services?"
    elif any(name in message_lower for name in ['mof', 'max life', 'lenovo', 'medanta', 'videocon']):
        response = f"That's one of our esteemed clients! We've provided various IT solutions including website development, ERP systems, and digital transformation services. Would you like to know more about the specific services we offer?"
    else:
        # Generic response for unknown project names
        response = f"That sounds like it could be one of our projects! We've worked on various IT solutions including websites, eCommerce platforms, ERP systems, and AI implementations. Would you like to know more about our services?"
    
    return create_button_response(response, [])

def is_blocked_domain(message: str) -> bool:
    """
    Check if the message contains a blocked domain that should be rejected
    """
    message_lower = message.lower().strip()
    
    blocked_domains = [
        'fascai.com', 'google.com', 'facebook.com', 'twitter.com', 'instagram.com',
        'linkedin.com', 'youtube.com', 'github.com', 'stackoverflow.com',
        'amazon.com', 'microsoft.com', 'apple.com', 'netflix.com', 'spotify.com',
        'wikipedia.org', 'reddit.com', 'pinterest.com', 'tiktok.com', 'snapchat.com',
        'flipkart.com', 'amazon.in', 'myntra.com', 'paytm.com', 'zomato.com',
        'swiggy.com', 'uber.com', 'ola.com', 'make my trip.com', 'booking.com'
    ]
    
    for blocked_domain in blocked_domains:
        if blocked_domain in message_lower:
            return True
    
    return False

def handle_blocked_domain() -> Dict:
    """
    Handle blocked domain queries with professional refusal message
    """
    refusal_message = "I specialize in Fasc Ai's solutions rather than other companies. However, I'd be happy to tell you how we compare! We've successfully delivered 250+ projects with cutting-edge IT solutions. What specific challenge can I help you solve?"
    
    return {
        'reply': refusal_message,
        'sources': [],
        'website_url': 'https://fascai.com'
    }

def is_business_info_query(message: str) -> bool:
    """
    Check if message is asking for business information
    """
    message_lower = message.lower().strip()
    
    business_keywords = [
        'team size', 'how many employees', 'how many people', 'team members',
        'location', 'where are you', 'office location', 'address',
        'certifications', 'certified', 'awards', 'recognition',
        'success rate', 'customer satisfaction', 'client satisfaction',
        'experience', 'years of experience', 'how long',
        'portfolio', 'case studies', 'projects completed',
        'clients', 'customer base', 'who are your clients'
    ]
    
    return any(keyword in message_lower for keyword in business_keywords)

# Removed hardcoded service detection functions - now handled by RAG flow with semantic matching
# Services are detected dynamically from ChromaDB based on distance thresholds (RELEVANCE_THRESHOLD = 1.5)
# If distance <= 1.5: Service found - provide information
# If distance > 1.5: Service not found - politely decline

def is_getting_started_query(message: str) -> bool:
    """
    Check if message is asking how to get started with Fasc AI
    """
    message_lower = message.lower().strip()
    
    getting_started_patterns = [
        'how do i get started', 'how to get started', 'how can i get started',
        'how do i start', 'how to start', 'how can i start',
        'how do i begin', 'how to begin', 'how can i begin',
        'get started', 'getting started', 'start working',
        'begin working', 'start a project', 'begin a project',
        'how to work with you', 'how to collaborate', 'how to partner',
        'how to engage', 'how to hire you', 'how to connect',
        'what is the first step', 'what are the next steps',
        'how to proceed', 'how to move forward'
    ]
    
    return any(pattern in message_lower for pattern in getting_started_patterns)

# Removed hardcoded getting started response - now handled by RAG flow

def is_company_info_query(message: str) -> bool:
    """
    Check if message is asking for company information
    """
    message_lower = message.lower().strip()
    
    # First check for getting started patterns - these should NOT be company info
    getting_started_patterns = [
        'how do i get started', 'how to get started', 'how can i get started',
        'how do i start', 'how to start', 'how can i start',
        'how do i begin', 'how to begin', 'how can i begin',
        'get started', 'getting started', 'start working',
        'begin working', 'start a project', 'begin a project'
    ]
    
    # If it's a getting started question, return False
    for pattern in getting_started_patterns:
        if pattern in message_lower:
            return False
    
    company_keywords = [
        'company size', 'business size', 'organization size',
        'founded', 'established', 'since when', 'started',
        'headquarters', 'main office', 'base location',
        'company culture', 'work environment', 'team culture',
        'company values', 'mission', 'vision', 'goals',
        'leadership', 'management', 'founder', 'ceo',
        'company history', 'background', 'story'
    ]
    
    return any(keyword in message_lower for keyword in company_keywords)

def generate_business_info_response(message: str) -> Dict:
    """
    Generate specific responses for business information queries
    """
    message_lower = message.lower().strip()
    
    # Team size queries
    if any(keyword in message_lower for keyword in ['team size', 'employees', 'people', 'team members']):
        return {
            'reply': "Our team consists of 50+ experienced professionals including developers, designers, project managers, and technical specialists. We have experts in cloud computing, AI, ERP, CRM, and IoT technologies.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Location queries
    elif any(keyword in message_lower for keyword in ['location', 'where are you', 'office', 'address']):
        return {
            'reply': "We are based in India with our main office serving clients globally. For specific contact details, email info@fascai.com or visit fascai.com/contact.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Certifications queries
    elif any(keyword in message_lower for keyword in ['certifications', 'certified', 'awards', 'recognition']):
        return {
            'reply': "Yes, Fasc Ai holds various industry-recognized certifications for IT solutions, cloud computing, ERP, CRM, AI solutions, and IoT implementations. We prioritize ongoing training to stay up-to-date with the latest technologies.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Success rate queries
    elif any(keyword in message_lower for keyword in ['success rate', 'customer satisfaction', 'client satisfaction']):
        return {
            'reply': "We maintain a 95%+ client satisfaction rate with over 250+ successfully delivered projects. Our clients include MOF, Max Life, Lenovo, Medanta, Videocon, and Saarte.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Experience queries
    elif any(keyword in message_lower for keyword in ['experience', 'years', 'how long']):
        return {
            'reply': "Fasc Ai has been operating for over 19 years. We've delivered 250+ projects across cloud computing, ERP, CRM, AI solutions, and IoT.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Portfolio queries
    elif any(keyword in message_lower for keyword in ['portfolio', 'case studies', 'projects', 'clients']):
        return {
            'reply': "We've delivered 250+ projects including Grand Trio Sports, Funzoop, Wonder Land Garden, Tysley, Dorundo, Dog Walking, and Matson Surgicals. Our clients include MOF, Max Life, Lenovo, Medanta, Videocon, and Saarte.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Default business info response
    else:
        return {
            'reply': "Fasc Ai Ventures Private Limited is a leading IT solutions company with 19+ years of experience, 50+ team members, and 250+ successfully delivered projects. We specialize in cloud computing, ERP, CRM, AI solutions, and IoT implementations.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }

def generate_company_info_response(message: str) -> Dict:
    """
    Generate specific responses for company information queries
    """
    message_lower = message.lower().strip()
    
    # Company size queries
    if any(keyword in message_lower for keyword in ['company size', 'business size', 'organization size']):
        return {
            'reply': "Fasc Ai Ventures Private Limited is a mid-sized IT solutions company with 50+ professionals. We serve clients ranging from startups to large enterprises and government organizations.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Founded/Established queries
    elif any(keyword in message_lower for keyword in ['founded', 'established', 'since when', 'started']):
        return {
            'reply': "Fasc Ai has been operating for over 19 years. We've grown from a small team to a company with 50+ professionals serving clients globally.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Headquarters queries
    elif any(keyword in message_lower for keyword in ['headquarters', 'main office', 'base location']):
        return {
            'reply': "Our main operations are based in India, serving clients globally. For specific office locations, email info@fascai.com or visit fascai.com/contact.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Company culture queries
    elif any(keyword in message_lower for keyword in ['company culture', 'work environment', 'team culture']):
        return {
            'reply': "At Fasc Ai, we foster a collaborative and innovative work environment. Our team culture emphasizes continuous learning, client satisfaction, and delivering cutting-edge IT solutions.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Company values queries
    elif any(keyword in message_lower for keyword in ['company values', 'mission', 'vision', 'goals']):
        return {
            'reply': "Our mission is to deliver innovative IT solutions that drive business success. We are committed to excellence, client satisfaction, and staying at the forefront of technology.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    
    # Default company info response
    else:
        return {
            'reply': "Fasc Ai Ventures Private Limited is a 19+ year old IT solutions company with 50+ professionals, specializing in cloud computing, ERP, CRM, AI solutions, and IoT implementations.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }

def is_contact_query(message: str) -> bool:
    """Check if message is asking for contact information"""
    message_lower = message.lower().strip()
    
    contact_keywords = [
        'how can i contact you', 'how to contact', 'contact information', 'contact details',
        'how to reach you', 'get in touch', 'contact you', 'reach you', 'phone number',
        'email address', 'contact method', 'how do i contact', 'contact support',
        'customer service', 'support contact', 'business contact'
    ]
    
    return any(keyword in message_lower for keyword in contact_keywords)

def is_technical_query(message: str) -> bool:
    """Check if message is asking about technical capabilities"""
    message_lower = message.lower().strip()
    
    technical_keywords = [
        'integrate with existing', 'system integration', 'api integration', 'database integration',
        'backup plans', 'backup strategy', 'disaster recovery', 'data backup',
        'technical support', 'technical capabilities', 'system compatibility',
        'integration capabilities', 'backup solutions', 'recovery plans',
        'can you integrate', 'do you integrate', 'integration support'
    ]
    
    return any(keyword in message_lower for keyword in technical_keywords)

def is_expertise_query(message: str) -> bool:
    """Check if message is asking about team expertise"""
    message_lower = message.lower().strip()
    
    expertise_keywords = [
        'team expertise', 'team skills', 'team capabilities', 'expertise areas',
        'what can you do', 'your capabilities', 'team experience', 'technical skills',
        'team knowledge', 'specializations', 'areas of expertise', 'team strengths',
        'what are you good at', 'your skills', 'team background'
    ]
    
    return any(keyword in message_lower for keyword in expertise_keywords)

def generate_contact_response(message: str) -> Dict:
    """Generate specific responses for contact queries"""
    return {
        'reply': "You can contact us at info@fascai.com or visit fascai.com/contact for detailed information.",
        'sources': [],
        'website_url': 'https://fascai.com'
    }

def generate_technical_response(message: str) -> Dict:
    """Generate specific responses for technical queries"""
    message_lower = message.lower().strip()
    
    if any(keyword in message_lower for keyword in ['integrate', 'integration', 'existing system']):
        return {
            'reply': "Yes, we specialize in system integration with APIs, databases, and third-party platforms. We integrate with existing systems seamlessly using modern techniques.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    elif any(keyword in message_lower for keyword in ['backup', 'backup plans', 'disaster recovery']):
        return {
            'reply': "Yes, we implement comprehensive backup strategies including data redundancy, disaster recovery, and business continuity plans.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }
    else:
        return {
            'reply': "Yes, we have extensive technical capabilities including system integration, backup solutions, API development, database management, and cloud infrastructure.",
            'sources': [],
            'website_url': 'https://fascai.com'
        }

def generate_expertise_response(message: str) -> Dict:
    """Generate specific responses for expertise queries"""
    return {
        'reply': "Our team has expertise in cloud computing, AI, ERP, CRM, IoT, web development, database design, automation, and system integration. We have 50+ professionals with 19+ years of experience.",
        'sources': [],
        'website_url': 'https://fascai.com'
    }

def is_timeline_query(message: str) -> bool:
    """Check if message is asking about project timelines or duration"""
    timeline_keywords = [
        'how long', 'how much time', 'timeline', 'duration', 'timeframe',
        'weeks', 'months', 'days', 'when will', 'how soon', 'delivery time',
        'project duration', 'completion time', 'how fast', 'quickly',
        'rush', 'urgent', 'asap', 'fast delivery', 'one month', '1 month',
        'two weeks', '2 weeks', 'three weeks', '3 weeks', 'few weeks'
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in timeline_keywords)

def generate_timeline_response(message: str) -> str:
    """Generate specific timeline responses based on the query"""
    message_lower = message.lower()
    
    # One month specific responses
    if any(phrase in message_lower for phrase in ['one month', '1 month', 'just one month']):
        responses = [
            "Absolutely! We can work within one month. We've delivered projects like Funzoop e-commerce in just 3 weeks.",
            "Yes, we can deliver within one month! We've completed projects like Wonder Land Garden in 2-3 weeks.",
            "One month is definitely doable! We've delivered multiple projects in 2-4 weeks."
        ]
        import random
        return random.choice(responses)
    
    # Website timeline responses
    if 'website' in message_lower:
        responses = [
            "We can build websites in 2-4 weeks for simple projects, or 1-2 months for complex ones. We've delivered Funzoop e-commerce in just 3 weeks.",
            "Website development typically takes 2-6 weeks depending on complexity. Simple sites: 2-3 weeks. E-commerce: 4-6 weeks. Complex applications: 6-8 weeks.",
            "For websites: Simple projects 2-3 weeks, e-commerce 4-6 weeks, complex applications 6-8 weeks. We've delivered Funzoop in just 3 weeks."
        ]
        import random
        return random.choice(responses)
    
    # General timeline responses
    responses = [
        "We work with flexible timelines. Simple projects: 2-4 weeks. Complex projects: 1-3 months.",
        "Timeline depends on project complexity. Simple projects: 2-4 weeks. Complex projects: 1-3 months. We've delivered projects like Funzoop in just 3 weeks.",
        "We can work with your timeline! Simple projects take 2-4 weeks, complex ones 1-3 months."
    ]
    import random
    return random.choice(responses)

def handle_existing_customer_query() -> Dict:
    """Handle existing customer queries and provide appropriate response with buttons"""
    import random
    
    responses = [
        "Welcome back! Great to see you again. How can I help you with your project today?",
        "Hello! Good to meet you again. What would you like to work on today?",
        "Hi there! Welcome back. How can I assist you with your project needs?"
    ]
    
    response_message = random.choice(responses)
    
    # Create buttons for existing customers
    buttons = [
        {"text": "Add New Features", "action": "add_features"},
        {"text": "Raise a Ticket", "action": "raise_ticket"}
    ]
    
    result = create_button_response(response_message, buttons)
    logger.info(f"handle_existing_customer_query returning: {result}")
    return result

def generate_existing_project_response(client_name: str, project_details: str = None) -> str:
    """Generate response for existing project recognition"""
    import random
    
    if project_details:
        responses = [
            f"Hi {client_name}! Great to see you again. I can see your project details in our system. How can I help you today?",
            f"Welcome back, {client_name}! I found your project information. What would you like to work on?",
            f"Hello {client_name}! Good to meet you again. I can see your project details. How can I assist you today?"
        ]
    else:
        responses = [
            f"Hi {client_name}! Good to see you again. How can I help you with your project today?",
            f"Welcome back, {client_name}! What would you like to work on today?",
            f"Hello {client_name}! Great to meet you again. How can I assist you?"
        ]
    
    return random.choice(responses)

def generate_project_not_found_response() -> str:
    """Generate response when project is not found"""
    import random
    
    responses = [
        "I couldn't find that project in our system. Could you please double-check the project name and client name? You can also start a new project if needed.",
        "Sorry, we couldn't locate that project. Please verify the project details and try again, or we can help you start a new project.",
        "I don't see that project in our records. Please check the project name and client information, or we can assist you with a new project."
    ]
    
    return random.choice(responses)

def generate_form_submission_response(form_type: str) -> str:
    """Generate response after form submission"""
    responses = {
        'new_project': [
            "Thank you! We've received your new project details. Our team will review your requirements and get back to you within 24 hours. We're excited to work with you!",
            "Excellent! Your project details have been submitted successfully. Our team will analyze your requirements and contact you within 24 hours.",
            "Perfect! We've received your project information. Our team will review it and get back to you soon with next steps."
        ],
        'existing_project': [
            "Thank you! We've received your existing project information. We'll verify the details and get back to you shortly.",
            "Great! Your project details have been submitted. We'll review the information and contact you soon.",
            "Perfect! We've received your project information. Our team will verify it and get back to you."
        ],
        'add_features': [
            "Thank you! We've received your feature request. Our development team will review your requirements and provide a timeline for implementation.",
            "Excellent! Your feature request has been submitted. Our team will analyze the requirements and get back to you with an implementation plan.",
            "Great! We've received your feature request. Our development team will review it and provide you with next steps."
        ],
        'raise_ticket': [
            "Thank you! We've received your support ticket. Our technical team will address your issue and get back to you within 24 hours.",
            "Perfect! Your support ticket has been submitted. Our technical team will review your issue and contact you soon.",
            "Excellent! We've received your support ticket. Our team will investigate the issue and get back to you within 24 hours."
        ],
        'complaint': [
            "Thank you for bringing this to our attention. We've received your complaint and will address it promptly. Our team will contact you within 24 hours.",
            "We appreciate you sharing your concerns. Your complaint has been submitted and our team will investigate and resolve this issue quickly.",
            "Thank you for your feedback. We've received your complaint and our team will address it immediately and get back to you soon."
        ]
    }
    
    import random
    form_responses = responses.get(form_type, ["Thank you! We've received your submission and will get back to you soon."])
    return random.choice(form_responses)

def get_form_url(form_type: str) -> str:
    """Get the appropriate form URL for the form type"""
    return FORM_URLS.get(form_type, FORM_URLS['new_project'])

def create_button_response(message: str, buttons: List[Dict[str, str]]) -> Dict:
    """
    Create response with buttons for interactive workflow
    """
    return {
        'reply': message,
        'buttons': buttons,
        'sources': [],
        'website_url': "https://fascai.com"
    }

def handle_project_workflow(message: str, search_function, session_context: dict = None) -> Optional[Dict]:
    """
    Main function to handle project management workflow
    """
    # Import the flag from the main app
    try:
        import app_chromadb
        if not app_chromadb.PROJECT_FEATURES_ENABLED:
            return None
    except:
        if not PROJECT_FEATURES_ENABLED:
            return None
    
    try:
        # Check for blocked domains FIRST - reject with professional message
        if is_blocked_domain(message):
            logger.info(f"Detected blocked domain: {message}")
            return handle_blocked_domain()
        
        # Check for contact queries - give specific responses
        if is_contact_query(message):
            logger.info(f"Detected contact query: {message}")
            return generate_contact_response(message)
        
        # Check for technical queries - give specific responses
        if is_technical_query(message):
            logger.info(f"Detected technical query: {message}")
            return generate_technical_response(message)
        
        # Check for expertise queries - give specific responses
        if is_expertise_query(message):
            logger.info(f"Detected expertise query: {message}")
            return generate_expertise_response(message)
        
        # Check for timeline queries - give specific responses instead of AI generation
        if is_timeline_query(message):
            logger.info(f"Detected timeline query: {message}")
            timeline_response = generate_timeline_response(message)
            return {
                'reply': timeline_response,
                'sources': [],
                'website_url': 'https://fascai.com',
                'buttons': [{'text': 'New Project', 'action': 'new_project'}, {'text': 'Existing Project', 'action': 'existing_project'}]
            }
        
        # Check for business information queries - give specific responses
        if is_business_info_query(message):
            logger.info(f"Detected business info query: {message}")
            return generate_business_info_response(message)
        
        # Removed hardcoded service detection - now handled by RAG flow with semantic matching
        # Services are detected dynamically from ChromaDB based on distance thresholds
        
        # Check for company information queries - give specific responses
        if is_company_info_query(message):
            logger.info(f"Detected company info query: {message}")
            return generate_company_info_response(message)
        
        # Check for projects inquiry first (general questions about projects)
        if is_projects_inquiry(message):
            logger.info(f"Detected projects inquiry: {message}")
            return handle_projects_inquiry()
        
        # Check for existing project query FIRST (when context indicates waiting for existing project)
        if is_existing_project_query(message, session_context, search_function):
            logger.info(f"Detected existing project query: {message}")
            
            # Clear the session context since we're handling the existing project query
            if session_context and session_context.get('waiting_for_existing_project', False):
                try:
                    import app_chromadb
                    # Find the session and clear the context
                    for session_id, session_data in app_chromadb.conversation_sessions.items():
                        if session_data.get('project_context', {}).get('waiting_for_existing_project', False):
                            session_data['project_context']['waiting_for_existing_project'] = False
                            logger.info(f"Cleared existing project context for session: {session_id}")
                            break
                except Exception as e:
                    logger.error(f"Error clearing session context: {e}")
            
            # STEP 1: WHITELIST CHECK ORIGINAL MESSAGE FIRST (100% accuracy)
            # Check original message against whitelist BEFORE extraction
            # This handles multi-word names like "max life" correctly
            if is_in_known_customers(message):
                logger.info(f"Original message '{message}' found in whitelist - allowing directly")
                # Known customer - use original message (title case) as client name
                # This preserves multi-word names like "Max Life", "Grand Trio", "Dog Walking"
                # instead of splitting them incorrectly
                cleaned_client_name = clean_project_name(message)
                
                # Allow directly with "Welcome back" response
                response_message = generate_existing_project_response(cleaned_client_name, None)
                buttons = [
                    {"text": "Add New Features", "action": "add_features"},
                    {"text": "Raise a Ticket", "action": "raise_ticket"}
                ]
                return create_button_response(response_message, buttons)
            
            # STEP 2: Extract project details if not matched in original message
            project_name, client_name = extract_project_details(message)
            
            if project_name and client_name:
                # Clean the client name before using it
                cleaned_client_name = clean_project_name(client_name)
                
                # STEP 3: Check extracted project_name against whitelist (fallback)
                if is_in_known_customers(project_name):
                    logger.info(f"Extracted project {project_name} found in whitelist - allowing directly")
                    # Known customer - allow directly with "Welcome back" response
                    response_message = generate_existing_project_response(cleaned_client_name, None)
                    buttons = [
                        {"text": "Add New Features", "action": "add_features"},
                        {"text": "Raise a Ticket", "action": "raise_ticket"}
                    ]
                    return create_button_response(response_message, buttons)
                
                # STEP 4: NOT IN WHITELIST - Politely decline
                else:
                    logger.info(f"Project {project_name} not in whitelist - politely declining")
                    decline_message = f"I'm not aware of any specific information about '{project_name}' related to Fasc Ai's services or IT solutions."
                    return create_button_response(decline_message, [])
            else:
                # Invalid project details
                response_message = "I couldn't extract the project details properly. Please provide your project details:\n\n• Project Name OR Project URL OR your Registered Name\n\nExample: 'fascai.com, Ravi Rajput'"
                buttons = [
                    {"text": "Try Again", "action": "existing_project"},
                    {"text": "New Project", "action": "new_project"}
                ]
                return create_button_response(response_message, buttons)
        
        # Check for single project showcase (specific project names for showcase)
        # BUT ONLY if user is NOT in existing project context
        if not (session_context and session_context.get('waiting_for_existing_project', False)):
            if is_single_project_showcase(message):
                logger.info(f"Detected single project showcase: {message}")
                return handle_single_project_showcase(message)
        
        # Check for existing customer query
        if is_existing_customer_query(message):
            logger.info(f"Detected existing customer query: {message}")
            return handle_existing_customer_query()
        
        # Check for project intent
        if is_project_intent(message):
            logger.info(f"Detected project intent: {message}")
            
            # Create buttons for project selection
            buttons = [
                {"text": "New Project", "action": "new_project"},
                {"text": "Existing Project", "action": "existing_project"}
            ]
            
            response_message = generate_project_intent_response(message)
            result = create_button_response(response_message, buttons)
            logger.info(f"Project intent response: {result}")
            return result
        
        
        return None
        
    except Exception as e:
        logger.error(f"Error in project workflow: {e}")
        return None

def handle_button_action(action: str, additional_data: str = None, session_id: str = None, conversation_sessions: dict = None) -> Optional[Dict]:
    """
    Handle button actions for project management
    """
    # Import the flag from the main app
    try:
        import app_chromadb
        if not app_chromadb.PROJECT_FEATURES_ENABLED:
            return None
    except:
        if not PROJECT_FEATURES_ENABLED:
            return None
    
    try:
        if action == "new_project":
            form_url = get_form_url('new_project')
            return {
                'reply': f"Great! I'm opening the new project form for you. Please fill out the details and we'll get back to you soon.",
                'form_url': form_url,
                'form_type': 'new_project',
                'sources': [],
                'website_url': "https://fascai.com"
            }
        
        elif action == "existing_project":
            # Set session context to indicate we're waiting for existing project details
            logger.info(f"Existing project button clicked for session: {session_id}")
            if session_id and conversation_sessions is not None:
                try:
                    logger.info(f"Available sessions: {list(conversation_sessions.keys())}")
                    if session_id in conversation_sessions:
                        if 'project_context' not in conversation_sessions[session_id]:
                            conversation_sessions[session_id]['project_context'] = {}
                        conversation_sessions[session_id]['project_context']['waiting_for_existing_project'] = True
                        logger.info(f"Set session context for existing project: {session_id}")
                        logger.info(f"Session context: {conversation_sessions[session_id]['project_context']}")
                        logger.info(f"Full session data keys: {list(conversation_sessions[session_id].keys())}")
                    else:
                        logger.error(f"Session {session_id} not found in conversation_sessions!")
                except Exception as e:
                    logger.error(f"Error setting session context: {e}", exc_info=True)
            else:
                if not session_id:
                    logger.error("No session_id provided to existing_project action!")
                if conversation_sessions is None:
                    logger.error("No conversation_sessions provided to handle_button_action!")
            
            return {
                'reply': "Please provide your project details:\n\n• Project Name OR Project URL OR your Registered Name\n\nExample: 'fascai.com, Ravi Rajput'",
                'sources': [],
                'website_url': "https://fascai.com"
            }
        
        elif action == "add_features":
            form_url = get_form_url('add_features')
            return {
                'reply': f"Perfect! I'm opening the feature request form for you. Please provide your project name and describe the new features you need.",
                'form_url': form_url,
                'form_type': 'add_features',
                'sources': [],
                'website_url': "https://fascai.com"
            }
        
        elif action == "raise_ticket":
            form_url = get_form_url('raise_ticket')
            return {
                'reply': f"Got it! I'm opening the support ticket form for you. Please provide your project details and describe the issue.",
                'form_url': form_url,
                'form_type': 'raise_ticket',
                'sources': [],
                'website_url': "https://fascai.com"
            }
        
        elif action == "complaint":
            form_url = get_form_url('complaint')
            return {
                'reply': f"I understand your concern. I'm opening a complaint form for you to provide more details.",
                'form_url': form_url,
                'form_type': 'complaint',
                'sources': [],
                'website_url': "https://fascai.com"
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error handling button action {action}: {e}")
        return None

def handle_form_submission(form_type: str) -> Dict:
    """
    Handle form submission and generate appropriate response
    """
    if not PROJECT_FEATURES_ENABLED:
        return {
            'reply': "Thank you for your submission.",
            'sources': [],
            'website_url': "https://fascai.com"
        }
    
    try:
        response_message = generate_form_submission_response(form_type)
        return {
            'reply': response_message,
            'sources': [],
            'website_url': "https://fascai.com"
        }
    except Exception as e:
        logger.error(f"Error handling form submission for {form_type}: {e}")
        return {
            'reply': "Thank you for your submission. We'll get back to you soon.",
            'sources': [],
            'website_url': "https://fascai.com"
        }

# Enable project features (call this function to enable)
def enable_project_features():
    """Enable project management features"""
    global PROJECT_FEATURES_ENABLED
    PROJECT_FEATURES_ENABLED = True
    logger.info("Project management features enabled")

# Disable project features (call this function to disable)
def disable_project_features():
    """Disable project management features"""
    global PROJECT_FEATURES_ENABLED
    PROJECT_FEATURES_ENABLED = False
    logger.info("Project management features disabled")

# Check if project features are enabled
def is_project_features_enabled() -> bool:
    """Check if project management features are enabled"""
    return PROJECT_FEATURES_ENABLED
