"""
Setup file for SatyaAI package
"""
from setuptools import setup, find_packages

setup(
    name="satyaai",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        # Add your project dependencies here
        'fastapi>=0.68.0',
        'uvicorn>=0.15.0',
        'python-multipart',
        'pytest',
        'pytest-cov',
    ],
    python_requires='>=3.8',
)
