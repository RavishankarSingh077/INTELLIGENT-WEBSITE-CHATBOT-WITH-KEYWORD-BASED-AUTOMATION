#!/usr/bin/env python3
"""
Test script to verify chatbot fixes
"""
import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"

def test_chatbot(query, session_id=None):
    """Test chatbot with a query"""
    url = f"{BASE_URL}/chat"
    payload = {
        "message": query,
        "session_id": session_id or "test_session"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("reply", "No response")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Connection error: {str(e)}"

def main():
    print("=" * 60)
    print("Testing Chatbot Fixes")
    print("=" * 60)
    
    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    time.sleep(2)
    
    # Test queries
    test_queries = [
        "do you provide hardware installation and maintenance services",
        "do you provide courier service",
        "do you provide computer vision services",
        "do you provide Email services",
        "hi",
        "how can i get rid of smoking"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {query}")
        print(f"{'='*60}")
        response = test_chatbot(query)
        print(f"Response: {response}")
        print(f"Response length: {len(response)} characters")
        time.sleep(1)  # Small delay between requests

if __name__ == "__main__":
    main()

