# website Professional Chatbot

A professional AI-powered chatbot for FASC.AI website that provides intelligent responses about company services and products using OpenAI's GPT models.

## 🚀 Features

- **AI-Powered Responses**: Uses OpenAI GPT-4o-mini for intelligent conversations
- **RAG System**: Automatically scrapes and learns from fascai.com content
- **Professional UI**: Light blue theme matching company branding
- **Real-time Chat**: Typing animations and smooth user experience
- **AWS Ready**: Designed for AWS EC2 deployment
- **Secure**: API keys protected on backend, CORS configured
- **Responsive**: Works on desktop and mobile devices

## 📁 Project Structure

```
fascai-chatbot/
├── backend/
│   ├── app.py                 # FastAPI application
│   ├── requirements.txt       # Python dependencies
│   └── utils/
│       └── rag_helper.py     # RAG system for website scraping
├── frontend/
│   └── chatbot.html          # Chat widget for website integration
└── README.md                 # This file
```

## 🛠️ Setup Instructions

### 1. Backend Setup (AWS EC2)

#### Prerequisites
- AWS EC2 instance (Ubuntu 22.04 LTS recommended)
- OpenAI API key
- Python 3.9+

#### Installation Steps

1. **Connect to your EC2 instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

2. **Update system and install dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv git -y
   ```

3. **Clone and setup project:**
   ```bash
   git clone <your-repo-url>
   cd fascai-chatbot/backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key-here"
   ```

5. **Run the application:**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

6. **Configure AWS Security Group:**
   - Open port 8000 for HTTP traffic
   - Optionally open port 443 for HTTPS

### 2. Frontend Integration

1. **Copy the chatbot code:**
   - Open `frontend/chatbot.html`
   - Copy the entire content

2. **Update API URL:**
   - Replace `https://your-backend-url.com/chat` with your actual EC2 URL
   - Example: `https://your-ec2-ip:8000/chat`

3. **Add to your website:**
   - Paste the code before `</body>` tag in your website's HTML
   - Or include it as a separate script file

### 3. Optional: Custom Domain Setup

1. **Using Cloudflare (Recommended):**
   - Add your EC2 IP to Cloudflare
   - Create subdomain: `api.fascai.com`
   - Enable SSL/TLS

2. **Using AWS Route 53:**
   - Create hosted zone for your domain
   - Add A record pointing to EC2 IP

## 🔧 Configuration

### Backend Configuration

Update `backend/app.py`:
```python
# Change allowed origins to your domain
allow_origins=["https://fascai.com", "https://www.fascai.com"]

# Update OpenAI model if needed
model="gpt-4o-mini"  # or "gpt-3.5-turbo" for faster/cheaper responses
```

### Frontend Configuration

Update `frontend/chatbot.html`:
```javascript
const FASC_AI_CONFIG = {
    apiUrl: 'https://api.fascai.com/chat', // Your backend URL
    maxRetries: 3,
    retryDelay: 1000
};
```

## 🚀 Deployment Options

### Option 1: AWS EC2 (Recommended for Production)
- **Pros**: Full control, scalable, professional
- **Cons**: Requires server management
- **Cost**: ~$5-20/month depending on instance size

### Option 2: Render.com (Easy Setup)
- **Pros**: Simple deployment, automatic scaling
- **Cons**: Limited free tier, slower cold starts
- **Cost**: Free tier available, $7/month for always-on

### Option 3: Railway.app
- **Pros**: Modern platform, easy deployment
- **Cons**: Limited free credits
- **Cost**: $5/month after free credits

## 📊 Performance Expectations

- **Response Time**: 1-4 seconds per message
- **Concurrent Users**: 10-50 (depending on EC2 instance)
- **Uptime**: 99%+ with proper AWS setup
- **Cost**: ~$0.01-0.05 per conversation (OpenAI API)

## 🔒 Security Features

- API keys stored securely on backend
- CORS configured for your domain only
- Input validation and error handling
- Rate limiting (can be added)
- HTTPS support with SSL certificates

## 🎨 Customization

### Changing Colors
Update CSS variables in `frontend/chatbot.html`:
```css
/* Change these colors to match your brand */
background: linear-gradient(135deg, #8ec5fc, #6fb1fc);
```

### Adding Company Context
Update `backend/utils/rag_helper.py` to scrape additional pages:
```python
def scrape_website(base_url="https://fascai.com"):
    # Add more URLs to scrape
    urls = ["https://fascai.com", "https://fascai.com/about", "https://fascai.com/services"]
    # ... scraping logic
```

## 🐛 Troubleshooting

### Common Issues

1. **CORS Errors:**
   - Check `allow_origins` in `app.py`
   - Ensure your domain is correctly listed

2. **OpenAI API Errors:**
   - Verify API key is correct
   - Check API usage limits
   - Ensure sufficient credits

3. **Website Scraping Issues:**
   - Check if fascai.com is accessible
   - Verify website structure hasn't changed
   - Check firewall/security group settings

4. **Slow Responses:**
   - Consider upgrading EC2 instance
   - Use gpt-3.5-turbo for faster responses
   - Check network latency

## 📈 Monitoring & Analytics

### Basic Monitoring
- Check EC2 instance health in AWS Console
- Monitor OpenAI API usage in OpenAI dashboard
- Use browser developer tools to check API responses

### Advanced Monitoring (Optional)
- Add logging to track user interactions
- Implement analytics dashboard
- Set up alerts for downtime

## 🔄 Updates & Maintenance

### Regular Tasks
- Update dependencies monthly
- Monitor OpenAI API costs
- Check website content changes
- Review security settings

### Scaling Considerations
- Upgrade EC2 instance for more users
- Implement load balancing for high traffic
- Add caching for better performance
- Consider CDN for global users

## 📞 Support

For issues or questions:
1. Check this README first
2. Review AWS EC2 logs
3. Test API endpoints manually
4. Check OpenAI API status

## 🎯 Next Steps

1. **Deploy backend** to AWS EC2
2. **Integrate frontend** into fascai.com
3. **Test thoroughly** with real users
4. **Monitor performance** and optimize
5. **Add features** like file uploads, multi-language support

---

**Ready to deploy?** Follow the setup instructions above and your professional chatbot will be live! 🚀

