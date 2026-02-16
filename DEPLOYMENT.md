# FASC.AI Chatbot - Production Deployment

## 🚀 Quick Start

### 1. Backend Deployment (AWS EC2)

```bash
# 1. Connect to your EC2 instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 2. Install dependencies
sudo apt update
sudo apt install python3-pip python3-venv git -y

# 3. Upload files to EC2 (use SCP or git clone)
# Upload: app.py, requirements.txt, utils/rag_helper.py

# 4. Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Set environment variable
export OPENAI_API_KEY="your-openai-api-key-here"

# 6. Run the server
python app.py
```

### 2. Frontend Integration

1. **Copy the widget code:**
   - Open `chatbot-widget.html`
   - Copy the entire content

2. **Update API URL:**
   - Replace `https://your-backend-url.com/chat` with your EC2 URL
   - Example: `https://your-ec2-ip:8000/chat`

3. **Add to your website:**
   - Paste the code before `</body>` tag in fascai.com

## 📁 Final File Structure

```
fascai-chatbot/
├── app.py                    # Main FastAPI application
├── requirements.txt          # Python dependencies
├── utils/
│   └── rag_helper.py        # RAG system for website scraping
├── chatbot-widget.html      # Frontend widget for website
└── DEPLOYMENT.md           # This file
```

## 🔧 Configuration

### Backend Configuration

Update `app.py` line 12:
```python
allow_origins=["https://fascai.com", "https://www.fascai.com"]
```

### Frontend Configuration

Update `chatbot-widget.html` line 347:
```javascript
apiUrl: 'https://your-ec2-ip:8000/chat'
```

## 🌐 Production Setup

### AWS Security Group
- Port 22: SSH access
- Port 8000: HTTP traffic
- Port 443: HTTPS (optional)

### Domain Setup (Optional)
- Use Cloudflare or Route 53
- Point `api.fascai.com` to your EC2 IP
- Enable SSL certificate

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## 🎯 Features

✅ **AI-Powered**: OpenAI GPT-4o-mini integration  
✅ **RAG System**: Automatically learns from fascai.com  
✅ **Professional UI**: Light blue company theme  
✅ **Mobile Ready**: Responsive design  
✅ **Fast Response**: Typing animations  
✅ **Secure**: API keys protected, CORS configured  

## 📊 Performance

- **Response Time**: 1-4 seconds per message
- **Concurrent Users**: 10-50 (depending on EC2 instance)
- **Cost**: ~$0.01-0.05 per conversation (OpenAI API)

## 🔍 Testing

### Backend Test
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### Frontend Test
- Open `chatbot-widget.html` in browser
- Try messages: "hello", "services", "contact", "about"

## 🚨 Troubleshooting

### Common Issues

1. **CORS Errors:**
   - Check `allow_origins` in `app.py`
   - Ensure your domain is listed

2. **OpenAI API Errors:**
   - Verify API key is correct
   - Check API usage limits

3. **Website Scraping Issues:**
   - Check if fascai.com is accessible
   - Verify firewall settings

## 📈 Monitoring

### Basic Monitoring
- Check EC2 instance health
- Monitor OpenAI API usage
- Use browser developer tools

### Logs
```bash
# View application logs
tail -f nohup.out

# Check system status
sudo systemctl status your-service
```

---

**Your FASC.AI Chatbot is ready for production!** 🚀

**Next Steps:**
1. Deploy backend to AWS EC2
2. Add frontend widget to fascai.com
3. Test thoroughly
4. Monitor performance
5. Go live! 💬










