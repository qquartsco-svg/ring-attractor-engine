"""
Ring Attractor Engine - Setup Configuration
링어트랙트 엔진 패키지 설정
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ring-attractor-engine",
    version="1.0.0",
    author="Qquarts co.",
    author_email="",
    description="Ring Attractor Engine - Persistent state engine for continuous memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qquartsco-svg/ring-attractor-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "neurons-engine>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/qquartsco-svg/ring-attractor-engine/issues",
        "Source": "https://github.com/qquartsco-svg/ring-attractor-engine",
        "Documentation": "https://github.com/qquartsco-svg/ring-attractor-engine#readme",
    },
)

