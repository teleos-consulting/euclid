from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# Additional dependencies for development and testing
dev_requirements = [
    "pytest>=7.3.1",
    "pytest-cov>=4.1.0",
    "black>=23.3.0",
    "isort>=5.12.0",
    "mypy>=1.3.0",
    "flake8>=6.0.0",
]

# RAG-specific dependencies
rag_requirements = [
    "sentence-transformers>=2.2.2",
    "scikit-learn>=1.3.0",
]

# Server dependencies
server_requirements = [
    "fastapi>=0.95.1",
    "uvicorn>=0.22.0",
]

# Web browsing dependencies
web_requirements = [
    "beautifulsoup4>=4.12.0",
    "html2text>=2020.1.16",
]

# Optional web search dependencies
search_requirements = [
    "serpapi>=0.1.0",  # Optional for search functionality
]

setup(
    name="euclid",
    version="0.1.0",
    author="Euclid Team",
    author_email="euclid@example.com",
    description="A CLI tool for interacting with local Ollama models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/euclid",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: User Interfaces",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "rag": rag_requirements,
        "server": server_requirements,
        "web": web_requirements,
        "search": search_requirements,
        "web-full": web_requirements + search_requirements,
        "all": dev_requirements + rag_requirements + server_requirements + web_requirements + search_requirements,
    },
    entry_points={
        "console_scripts": [
            "euclid=euclid.tui:run_tui",
            "euclid-cli=euclid.cli:app",
        ],
    },
    include_package_data=True,
    package_data={
        "euclid": ["prompts/*.txt"],
    },
)