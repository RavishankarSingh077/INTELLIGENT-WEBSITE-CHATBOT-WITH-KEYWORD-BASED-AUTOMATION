# AWS EC2 Deployment Script

This script automates the deployment of FASC.AI Chatbot on AWS EC2.

## Prerequisites

1. AWS EC2 instance running Ubuntu 22.04 LTS
2. SSH access to your EC2 instance
3. OpenAI API key

## Quick Deployment

### Step 1: Connect to EC2
```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### Step 2: Run Deployment Script
```bash
# Download and run the deployment script
curl -o deploy.sh https://raw.githubusercontent.com/your-repo/fascai-chatbot/main/deploy.sh
chmod +x deploy.sh
./deploy.sh
```

### Step 3: Set Environment Variables
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### Step 4: Start the Application
```bash
cd fascai-chatbot/backend
source venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Manual Deployment Steps

### 1. System Update
```bash
sudo apt update
sudo apt upgrade -y
sudo apt install python3-pip python3-venv git nginx -y
```

### 2. Clone Repository
```bash
git clone https://github.com/your-username/fascai-chatbot.git
cd fascai-chatbot/backend
```

### 3. Setup Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Create environment file
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

### 5. Test Application
```bash
# Test the application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Setup Systemd Service (Optional)
```bash
# Create systemd service file
sudo nano /etc/systemd/system/fascai-chatbot.service
```

Add the following content:
```ini
[Unit]
Description=FASC.AI Chatbot API
After=network.target

[Service]
Type=exec
User=ubuntu
WorkingDirectory=/home/ubuntu/fascai-chatbot/backend
Environment=PATH=/home/ubuntu/fascai-chatbot/backend/venv/bin
ExecStart=/home/ubuntu/fascai-chatbot/backend/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable fascai-chatbot
sudo systemctl start fascai-chatbot
sudo systemctl status fascai-chatbot
```

### 7. Configure Nginx (Optional)
```bash
# Create nginx configuration
sudo nano /etc/nginx/sites-available/fascai-chatbot
```

Add the following content:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/fascai-chatbot /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 8. Configure Firewall
```bash
# Allow HTTP and HTTPS traffic
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 8000
sudo ufw --force enable
```

## AWS Security Group Configuration

Ensure your EC2 security group allows:
- **Port 22**: SSH access
- **Port 80**: HTTP traffic
- **Port 443**: HTTPS traffic
- **Port 8000**: Direct API access (optional)

## SSL Certificate Setup (Optional)

### Using Let's Encrypt
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Monitoring & Logs

### View Application Logs
```bash
# If using systemd service
sudo journalctl -u fascai-chatbot -f

# If running manually
tail -f nohup.out
```

### Check Application Status
```bash
# Test API endpoint
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

## Troubleshooting

### Common Issues

1. **Permission Denied:**
   ```bash
   sudo chown -R ubuntu:ubuntu /home/ubuntu/fascai-chatbot
   ```

2. **Port Already in Use:**
   ```bash
   sudo lsof -i :8000
   sudo kill -9 <PID>
   ```

3. **Python Module Not Found:**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **OpenAI API Errors:**
   - Check API key is correct
   - Verify API credits
   - Check rate limits

## Performance Optimization

### 1. Use Production WSGI Server
```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 2. Enable Caching
```bash
# Install Redis for caching
sudo apt install redis-server -y
pip install redis
```

### 3. Database Optimization
```bash
# Optimize ChromaDB
# The vector database will be automatically optimized on first run
```

## Backup & Recovery

### Backup Script
```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
tar -czf fascai-chatbot-backup-$DATE.tar.gz /home/ubuntu/fascai-chatbot
aws s3 cp fascai-chatbot-backup-$DATE.tar.gz s3://your-backup-bucket/
```

### Recovery
```bash
# Download backup
aws s3 cp s3://your-backup-bucket/fascai-chatbot-backup-YYYYMMDD_HHMMSS.tar.gz ./

# Extract backup
tar -xzf fascai-chatbot-backup-YYYYMMDD_HHMMSS.tar.gz

# Restore application
sudo systemctl stop fascai-chatbot
# Replace files
sudo systemctl start fascai-chatbot
```

## Scaling Considerations

### Horizontal Scaling
- Use Application Load Balancer
- Deploy multiple EC2 instances
- Use Auto Scaling Groups

### Vertical Scaling
- Upgrade EC2 instance type
- Increase memory and CPU
- Use RDS for database (if needed)

---

**Your FASC.AI Chatbot is now ready for production!** 🚀










