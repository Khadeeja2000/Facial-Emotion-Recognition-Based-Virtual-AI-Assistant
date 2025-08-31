"""
Setup script for Emotion Recognition System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="emotion-recognition",
    version="1.2.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive real-time emotion recognition platform using computer vision and deep learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/emotion-recognition",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/emotion-recognition/issues",
        "Source": "https://github.com/yourusername/emotion-recognition",
        "Documentation": "https://github.com/yourusername/emotion-recognition/wiki",
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
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.13.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "emotion-recognition=real_time_video:main",
            "train-emotion-model=train_emotion_classifier:train_model",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.xml", "*.hdf5", "*.h5"],
    },
    keywords="emotion recognition, computer vision, deep learning, facial expressions, AI, machine learning",
    license="MIT",
    zip_safe=False,
)
