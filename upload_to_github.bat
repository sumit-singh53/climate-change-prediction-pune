@echo off
echo 🚀 GitHub Upload Helper Script
echo ================================
echo.
echo This script will help you upload your project to GitHub
echo.
echo BEFORE RUNNING THIS SCRIPT:
echo 1. Create a new repository on GitHub.com named: climate-change-prediction-pune
echo 2. Make it PUBLIC (recommended)
echo 3. DO NOT initialize with README, .gitignore, or license
echo 4. Replace YOUR_USERNAME below with your actual GitHub username
echo.
pause
echo.

echo 📡 Adding GitHub remote...
git remote add origin https://github.com/sumit-singh53/climate-change-prediction-pune.git

echo 📤 Pushing to GitHub...
git push -u origin main

echo.
echo ✅ Upload complete!
echo.
echo 🌟 Next steps:
echo 1. Go to your repository on GitHub
echo 2. Add topics: climate-change, air-quality, iot, machine-learning, pune, streamlit, python
echo 3. Enable GitHub Pages in Settings
echo 4. Update README.md links with your username
echo.
pause