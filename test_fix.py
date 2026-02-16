import requests
import time

questions = [
    "What is ERP and how can you help with it?",
    "Can you help me set up a CRM system?",
    "Tell me about your cloud hosting solutions",
    "What services does FASC.AI offer?",
    "Do you provide IoT solutions?",
    "How much does a custom web application cost?"
]

print("\n" + "="*70)
print("CHATBOT TEST - After Pattern Check Fix")
print("="*70 + "\n")

for i, q in enumerate(questions, 1):
    try:
        # Wait a bit to avoid rate limiting
        if i > 1:
            time.sleep(2)
        
        resp = requests.post(
            "http://localhost:8000/chat",
            json={"message": q},
            timeout=20
        )
        reply = resp.json()["reply"]
        print(f"[Q{i}] {q}")
        print(f"[R{i}] {reply}")
        print("-"*70 + "\n")
    except Exception as e:
        print(f"[Q{i}] {q}")
        print(f"[ERROR] {str(e)}")
        print("-"*70 + "\n")


