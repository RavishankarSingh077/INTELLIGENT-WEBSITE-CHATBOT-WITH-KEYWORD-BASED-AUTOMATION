# FASC.AI Chatbot Deployment Playbook

This guide explains how to deploy the FastAPI + Groq-powered chatbot on AWS EC2 with Gunicorn + Nginx, and how to embed the existing front-end widget on any website.

---

## 1. Provision & Prepare the EC2 Host

1. **Launch instance**
   - AWS EC2 → Launch Ubuntu 22.04 LTS (or Amazon Linux 2).
   - Instance type: `t2.medium` or larger (for embedding + LLM calls).
   - Security Group: open inbound ports **22** (SSH) and **80** (HTTP). HTTPS will be added later.

2. **SSH into the instance**
   ```bash
   ssh -i your-key.pem ubuntu@EC2_PUBLIC_IP
   ```

3. **Install OS prerequisites**
   ```bash
   sudo apt update
   sudo apt install -y git python3.10 python3-pip python3-venv nginx
   # optional but recommended for some Python wheels
   sudo apt install -y build-essential
   ```

---

## 2. Pull Code & Create Python Environment

```bash
git clone https://github.com/FASC-AI/AI.git -b Chatbot_Agent
cd AI
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Configure Environment Variables

- Export your Groq API keys (comma-separated) and any other secrets:
  ```bash
  export GROQ_API_KEYS="key1,key2"
  export CHROMA_DB_PATH="/home/ubuntu/AI/chroma_db"
  ```
  > You can also place them inside an `.env` file and load with `dotenv`.

- Ensure the local Chroma storage directory exists:
  ```bash
  mkdir -p chroma_db
  ```

---

## 4. Smoke-Test the Backend

```bash
uvicorn app_chromadb:app --host 0.0.0.0 --port 8000
```

- In another terminal or with `curl`, hit:
  ```bash
  curl http://EC2_PUBLIC_IP:8000/health
  ```
  Expect: `{"status":"healthy"}`.
- Stop Uvicorn (`Ctrl+C`) after confirming.

---

## 5. Run via Gunicorn (manual check)

```bash
gunicorn app_chromadb:app \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

- Test `http://EC2_PUBLIC_IP:8000/chat` with a POST request.
- Stop (`Ctrl+C`) once verified.

---

## 6. Systemd Service (keep Gunicorn running)

1. **Create service file**
   ```bash
   sudo nano /etc/systemd/system/chatbot.service
   ```

   ```ini
   [Unit]
   Description=FASC.AI Chatbot
   After=network.target

   [Service]
   User=ubuntu
   WorkingDirectory=/home/ubuntu/AI
   Environment="GROQ_API_KEYS=key1,key2"
   Environment="CHROMA_DB_PATH=/home/ubuntu/AI/chroma_db"
   ExecStart=/home/ubuntu/AI/venv/bin/gunicorn app_chromadb:app -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8000
   Restart=always
   RestartSec=5

  [Install]
   WantedBy=multi-user.target
   ```

   > Adjust `User`, paths, and environment variables to match your setup.

2. **Enable service**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl start chatbot
   sudo systemctl enable chatbot
   sudo systemctl status chatbot
   ```
   Status should show **active (running)**.

---

## 7. Nginx Reverse Proxy

1. **Create Nginx site**
   ```bash
   sudo nano /etc/nginx/sites-available/chatbot
   ```

   ```nginx
   server {
       listen 80;
       server_name yourdomain.com www.yourdomain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }

       location /static/chatbot/ {
           alias /var/www/chatbot/;
       }
   }
   ```

   > `alias` assumes you will host frontend assets under `/var/www/chatbot/`. Adjust to match your desired path.

2. **Enable site**
   ```bash
   sudo ln -s /etc/nginx/sites-available/chatbot /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

---

## 8. HTTPS (Recommended)

1. Point your domain’s **A record** to the EC2 public IP.
2. Install Certbot and issue certificates:
   ```bash
   sudo apt install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
   ```
3. Confirm `https://yourdomain.com/chat` returns the API response.

---

## 9. Host & Integrate the Frontend Widget

### 9.1 Host existing assets

The repo includes the ready-made widget (`frontend/project_management.js`, `frontend/project_management.css`, optional HTML demos). Serve them statically:

```bash
sudo mkdir -p /var/www/chatbot
sudo cp frontend/project_management.js frontend/project_management.css /var/www/chatbot/
```

Ensure `project_management.js` points to the public API URL (update any hard-coded `/chat` endpoint to the full domain if needed).

Restart Nginx if files change.

### 9.2 Drop-in snippet for any website

Add a placeholder div where the widget should appear:
```html
<div id="chatbot-box"></div>
```

Paste this snippet right before the closing `</body>` tag (over HTTPS):

```html
<!-- FASC.AI Chatbot -->
<link rel="stylesheet" href="https://yourdomain.com/static/chatbot/project_management.css">
<script src="https://yourdomain.com/static/chatbot/project_management.js" defer onload="
  window.FASC_AI && window.FASC_AI.init({
    apiUrl: 'https://yourdomain.com/chat',
    position: 'bottom-right',
    theme: 'light'
  });
"></script>
```

- `project_management.js` already sets up the UI, handles fetch calls, and manages sessions.
- Adjust `position` or `theme` as needed (options already supported in the JS).

---

## 10. Quick Checklist

- [ ] EC2 instance running with security groups for 22/80 (443 after TLS).
- [ ] Repo cloned, virtualenv installed, dependencies installed.
- [ ] Environment variables configured (Groq keys, DB paths).
- [ ] Gunicorn + systemd service active on `127.0.0.1:8000`.
- [ ] Nginx proxy serving on `80/443`, static assets reachable under `/static/chatbot/`.
- [ ] TLS certificate issued via Certbot (if using HTTPS).
- [ ] Frontend snippet added to website pages; widget loads and can call `https://yourdomain.com/chat`.

---

## 11. Maintenance Tips

- Rotate Groq API keys as needed; update `GROQ_API_KEYS` in the systemd service, then `sudo systemctl restart chatbot`.
- Monitor logs:
  ```bash
  journalctl -u chatbot -f
  tail -f /var/log/nginx/access.log
  tail -f /var/log/nginx/error.log
  ```
- Backup and clean `chroma_db` periodically if it grows large.
- Update the app:
  ```bash
  cd /home/ubuntu/AI
  git pull
  source venv/bin/activate
  pip install -r requirements.txt
  sudo systemctl restart chatbot
  ```

---

With this setup, the chatbot backend is production-ready, Gunicorn can serve multiple concurrent users, and the existing widget can be embedded on any site with a single snippet.



