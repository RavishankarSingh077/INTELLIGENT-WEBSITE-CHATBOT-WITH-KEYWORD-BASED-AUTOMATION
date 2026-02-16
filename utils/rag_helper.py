import requests
from bs4 import BeautifulSoup
import logging
from typing import Optional, List
import os
import hashlib
from urllib.parse import urljoin, urlparse

# Try to import ChromaDB and SentenceTransformers with fallback
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available, using fallback mode")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("SentenceTransformers not available, using fallback mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ChromaDB and SentenceTransformers initialization
def get_chroma_client():
    """Initialize ChromaDB client"""
    if not CHROMADB_AVAILABLE:
        return None
    try:
        client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        return client
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {str(e)}")
        return None

def get_embedding_model():
    """Initialize sentence transformer model"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        return None

def _extract_internal_links(soup, base_url: str) -> List[str]:
    """Extract internal links from a webpage"""
    domain = urlparse(base_url).netloc
    links = set()
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        absolute_url = urljoin(base_url, href)
        parsed = urlparse(absolute_url)
        
        # Only include internal links (same domain)
        if parsed.netloc == domain or parsed.netloc == '':
            # Remove fragments and query params for cleaner URLs
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if clean_url != base_url and clean_url not in links:
                links.add(clean_url)
    
    return list(links)

def scrape_website(url: str = "https://fascai.com", max_depth: Optional[int] = None) -> Optional[str]:
    """Scrape content from a website, optionally with crawling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        visited = set()
        pages_to_visit = [(url, 0)]  # (url, depth)
        all_content = []
        
        while pages_to_visit:
            current_url, depth = pages_to_visit.pop(0)
            
            # Skip if already visited or exceeds max depth
            if current_url in visited or (max_depth is not None and depth > max_depth):
                continue
            
            try:
                logger.info(f"Scraping: {current_url} (depth: {depth})")
                response = requests.get(current_url, headers=headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text from relevant elements
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
                
                # Get div content (if it contains substantial text)
                for tag in soup.find_all('div'):
                    text = tag.get_text().strip()
                    if len(text) > 50:  # Only include divs with substantial content
                        text_elements.append(text)
                
                # Clean and join text
                cleaned_text = ' '.join([text for text in text_elements if text])
                
                if cleaned_text:
                    all_content.append(f"--- Content from {current_url} ---")
                    all_content.append(cleaned_text)
                    logger.info(f"Successfully scraped {len(cleaned_text)} characters from {current_url}")
                
                visited.add(current_url)
                
                # If crawling is enabled, find and add internal links
                if max_depth is not None and depth < max_depth:
                    internal_links = _extract_internal_links(soup, current_url)
                    for link in internal_links[:10]:  # Limit to 10 links per page to avoid overload
                        if link not in visited:
                            pages_to_visit.append((link, depth + 1))
                            logger.info(f"Found internal link: {link}")
                
            except Exception as e:
                logger.error(f"Error scraping {current_url}: {str(e)}")
                continue
        
        # Combine all scraped content
        combined_content = '\n\n'.join(all_content)
        logger.info(f"Total pages scraped: {len(visited)}")
        logger.info(f"Total content length: {len(combined_content)} characters")
        
        return combined_content if combined_content else None
        
    except Exception as e:
        logger.error(f"Error in website scraping: {str(e)}")
        return None

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def store_content_in_vector_db(website_url: str = "https://fascai.com"):
    """Store website content in ChromaDB"""
    try:
        client = get_chroma_client()
        model = get_embedding_model()
        
        if not client or not model:
            logger.warning("ChromaDB or embedding model not available, using fallback mode")
            return False
        
        # Get or create collection
        collection_name = "fascai_content"
        try:
            collection = client.get_collection(name=collection_name)
            logger.info("Using existing collection")
        except:
            collection = client.create_collection(name=collection_name)
            logger.info("Created new collection")
        
        # Scrape website content
        content = scrape_website(website_url)
        if not content:
            return False
        
        # Chunk the content
        chunks = chunk_text(content)
        logger.info(f"Created {len(chunks)} chunks from website content")
        
        # Generate embeddings
        embeddings = model.encode(chunks)
        
        # Create unique IDs for each chunk
        chunk_ids = [f"chunk_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}" for i, chunk in enumerate(chunks)]
        
        # Store in ChromaDB
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=chunk_ids,
            metadatas=[{"source": website_url, "chunk_index": i} for i in range(len(chunks))]
        )
        
        logger.info(f"Successfully stored {len(chunks)} chunks in ChromaDB")
        return True
        
    except Exception as e:
        logger.error(f"Error storing content in ChromaDB: {str(e)}")
        return False

def search_vector_db(query: str, n_results: int = 3) -> List[str]:
    """Search ChromaDB for relevant content"""
    try:
        client = get_chroma_client()
        model = get_embedding_model()
        
        if not client or not model:
            logger.warning("ChromaDB or embedding model not available, using fallback mode")
            return []
        
        # Get collection
        collection_name = "fascai_content"
        try:
            collection = client.get_collection(name=collection_name)
        except:
            logger.warning("Collection not found, initializing with website content")
            if store_content_in_vector_db():
                collection = client.get_collection(name=collection_name)
            else:
                return []
        
        # Generate query embedding
        query_embedding = model.encode([query])
        
        # Search
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        # Extract documents
        documents = results.get('documents', [[]])[0]
        logger.info(f"Found {len(documents)} relevant documents for query: {query}")
        
        return documents
        
    except Exception as e:
        logger.error(f"Error searching ChromaDB: {str(e)}")
        return []

def get_rag_response(message):
    """Get RAG response using ChromaDB vector search"""
    try:
        # Search ChromaDB for relevant content
        relevant_docs = search_vector_db(message, n_results=3)
        
        if relevant_docs:
            # Combine relevant documents
            context = " ".join(relevant_docs)
            logger.info("Using ChromaDB vector search results for context")
            return context
        else:
            # Fallback to static content if no results
            logger.info("No ChromaDB results, using static content")
            return """
            Fasc AI Ventures Private Limited is a premier AI & IT solutions company specializing in:
            Cloud Transformation, Enterprise Resource Planning (ERP), Customer Relationship Management (CRM), 
            E-commerce & CMS solutions, Internet of Things (IoT), Mobility Solutions, Data & Analytics, 
            Web Design & Development, and AI Solutions. We have successfully implemented 250+ projects 
            for our esteemed clients including MOF, Max Life, Lenovo, Medanta, and more. Visit https://fascai.com for more information.
            """
            
    except Exception as e:
        logger.error(f"Error in RAG response: {str(e)}")
        return "Fasc AI Ventures Private Limited provides comprehensive IT solutions including Cloud Transformation, ERP, CRM, E-commerce, IoT, and AI solutions. Visit https://fascai.com for more information."

# Initialize vector database on module import (disabled - handled by main app)
def initialize_vector_db():
    """Initialize ChromaDB with website content - called manually if needed"""
    try:
        if CHROMADB_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            store_content_in_vector_db()
            logger.info("ChromaDB vector database initialized successfully")
        else:
            logger.info("Using fallback mode - ChromaDB/SentenceTransformers not available")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {str(e)}")

# Auto-initialization disabled to prevent duplicate initialization
# Call initialize_vector_db() manually if needed