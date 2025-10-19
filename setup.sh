#!/bin/bash

# Streamlit deployment setup script
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = \$PORT\n\
" > ~/.streamlit/config.toml

# Create necessary directories
mkdir -p data
mkdir -p logs
mkdir -p outputs
mkdir -p outputs/models
mkdir -p outputs/reports
mkdir -p backups

# Set permissions
chmod +x run_dashboard.py
chmod +x demo_dashboard.py
chmod +x test_dashboard.py

echo "Setup completed successfully!"