#!/usr/bin/env python3
"""
Setup script for Enhanced Climate & AQI Prediction System
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="climate-change-prediction-pune",
    version="2.0.0",
    author="Sumit Singh",
    author_email="sumit.singh@example.com",
    description="Enhanced Climate & AQI Prediction System for Pune with IoT integration",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/sumit-singh53/climate-change-prediction-pune",
    project_urls={
        "Bug Tracker": "https://github.com/sumit-singh53/climate-change-prediction-pune/issues",
        "Documentation": "https://github.com/sumit-singh53/climate-change-prediction-pune/blob/main/README.md",
        "Source Code": "https://github.com/sumit-singh53/climate-change-prediction-pune",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Internet of Things",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "flake8>=6.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "climate-prediction=run_system:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "climate-change",
        "air-quality",
        "iot",
        "machine-learning",
        "pune",
        "environmental-monitoring",
        "prediction",
        "streamlit",
        "dashboard",
    ],
)