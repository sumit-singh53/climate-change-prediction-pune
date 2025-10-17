# Deployment Guide

This guide covers different deployment options for the Enhanced Climate & AQI Prediction System.

## 🚀 Deployment Options

### 1. Local Development
### 2. Docker Deployment
### 3. Cloud Deployment (AWS/GCP/Azure)
### 4. Production Setup

---

## 🏠 Local Development

### Prerequisites
- Python 3.8+
- Git
- 4GB+ RAM
- 2GB+ disk space

### Quick Start
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/climate-change-prediction-pune.git
cd climate-change-prediction-pune

# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run system
python run_system.py
```

### Access Points
- **Dashboard**: http://localhost:8501
- **IoT API**: http://localhost:5000
- **MQTT**: localhost:1883

---

## 🐳 Docker Deployment

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/external data/iot \
    outputs/models outputs/logs outputs/figures

# Expose ports
EXPOSE 8501 5000 1883

# Run application
CMD ["python", "run_system.py", "--mode", "full"]
```

### Docker Compose
```yaml
version: '3.8'

services:
  climate-prediction:
    build: .
    ports:
      - "8501:8501"  # Streamlit dashboard
      - "5000:5000"  # IoT API
      - "1883:1883"  # MQTT
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped

  mosquitto:
    image: eclipse-mosquitto:2
    ports:
      - "1883:1883"
      - "9001:9001"
    volumes:
      - ./mosquitto.conf:/mosquitto/config/mosquitto.conf
    restart: unless-stopped

volumes:
  data:
  outputs:
```

### Build and Run
```bash
# Build image
docker build -t climate-prediction .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## ☁️ Cloud Deployment

### AWS Deployment

#### EC2 Instance
```bash
# Launch EC2 instance (t3.medium recommended)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo usermod -a -G docker ec2-user

# Clone and deploy
git clone https://github.com/YOUR_USERNAME/climate-change-prediction-pune.git
cd climate-change-prediction-pune
docker-compose up -d
```

#### ECS Deployment
```json
{
  "family": "climate-prediction",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "climate-prediction",
      "image": "YOUR_ACCOUNT.dkr.ecr.REGION.amazonaws.com/climate-prediction:latest",
      "portMappings": [
        {"containerPort": 8501, "protocol": "tcp"},
        {"containerPort": 5000, "protocol": "tcp"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/climate-prediction",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Platform

#### Cloud Run
```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: climate-prediction
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containers:
      - image: gcr.io/PROJECT_ID/climate-prediction
        ports:
        - containerPort: 8501
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
        env:
        - name: PORT
          value: "8501"
```

#### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: climate-prediction
spec:
  replicas: 2
  selector:
    matchLabels:
      app: climate-prediction
  template:
    metadata:
      labels:
        app: climate-prediction
    spec:
      containers:
      - name: climate-prediction
        image: gcr.io/PROJECT_ID/climate-prediction:latest
        ports:
        - containerPort: 8501
        - containerPort: 5000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: climate-prediction-service
spec:
  selector:
    app: climate-prediction
  ports:
  - name: dashboard
    port: 80
    targetPort: 8501
  - name: api
    port: 5000
    targetPort: 5000
  type: LoadBalancer
```

---

## 🏭 Production Setup

### Environment Variables
```bash
# .env file
DATABASE_URL=postgresql://user:pass@localhost:5432/climate_db
REDIS_URL=redis://localhost:6379
API_KEY_SECRET=your-secret-key
MQTT_BROKER_URL=mqtt://broker.example.com:1883
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### Nginx Configuration
```nginx
upstream climate_dashboard {
    server 127.0.0.1:8501;
}

upstream climate_api {
    server 127.0.0.1:5000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://climate_dashboard;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://climate_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Systemd Service
```ini
[Unit]
Description=Climate Prediction System
After=network.target

[Service]
Type=simple
User=climate
WorkingDirectory=/opt/climate-prediction
Environment=PATH=/opt/climate-prediction/.venv/bin
ExecStart=/opt/climate-prediction/.venv/bin/python run_system.py --mode full
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Database Setup (PostgreSQL)
```sql
-- Create database
CREATE DATABASE climate_prediction;
CREATE USER climate_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE climate_prediction TO climate_user;

-- Create tables (automated by application)
-- Tables will be created automatically on first run
```

---

## 📊 Monitoring & Logging

### Prometheus Metrics
```python
# Add to your application
from prometheus_client import Counter, Histogram, Gauge

sensor_data_received = Counter('sensor_data_total', 'Total sensor data points received')
prediction_requests = Counter('prediction_requests_total', 'Total prediction requests')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Climate Prediction System",
    "panels": [
      {
        "title": "Sensor Data Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(sensor_data_total[5m])"
          }
        ]
      }
    ]
  }
}
```

### Log Configuration
```yaml
version: 1
formatters:
  default:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: default
  file:
    class: logging.handlers.RotatingFileHandler
    filename: /var/log/climate-prediction/app.log
    maxBytes: 10485760
    backupCount: 5
    level: INFO
    formatter: default
loggers:
  '':
    level: INFO
    handlers: [console, file]
```

---

## 🔒 Security Considerations

### API Security
- Implement API key authentication
- Use HTTPS in production
- Rate limiting
- Input validation
- SQL injection prevention

### Network Security
- Firewall configuration
- VPN for MQTT access
- SSL/TLS for all connections
- Regular security updates

### Data Security
- Encrypt sensitive data
- Regular backups
- Access logging
- Data retention policies

---

## 🔧 Troubleshooting

### Common Issues

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8501
lsof -i :8501

# Kill process using port
sudo kill -9 $(lsof -t -i:8501)
```

#### Memory Issues
```bash
# Monitor memory usage
htop
free -h

# Increase swap space
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Database Connection
```bash
# Check database status
sudo systemctl status postgresql
sudo systemctl restart postgresql

# Test connection
psql -h localhost -U climate_user -d climate_prediction
```

---

## 📈 Scaling

### Horizontal Scaling
- Load balancer configuration
- Multiple application instances
- Database replication
- Caching layer (Redis)

### Vertical Scaling
- Increase CPU/RAM
- SSD storage
- Database optimization
- Model optimization

---

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t climate-prediction .
    
    - name: Deploy to production
      run: |
        docker save climate-prediction | gzip | ssh user@server 'gunzip | docker load'
        ssh user@server 'docker-compose up -d'
```

This deployment guide provides comprehensive options for running your climate prediction system in various environments, from local development to production cloud deployments.