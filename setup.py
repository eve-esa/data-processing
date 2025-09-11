from setuptools import setup, find_packages

setup(
    name = "eve",
    version = "0.1.0",
    packages = find_packages(),
    install_requires=[
        "aiofiles",
        "aiohttp",
        "loguru",
        "pydantic",
        "PyYAML", 
        "trafilatura",
        "rapidfuzz>=3.0.0",
        "nltk>=3.8",
        "datasketch>=1.6.0",
        "pdflatex>=0.1.3",
        "pytest-asyncio==1.1.0",
    ],
    extras_require={
        "pii": [
            "presidio-analyzer>=2.2.0",
            "presidio-anonymizer>=2.2.0",
            "flair>=0.15.0", 
            "spacy>=3.8.0",
            "litserve",
        ],
        "metadata": [
            "pdf2bib",
            "pony",
            "python-dotenv",
        ],
        "all": [
            "presidio-analyzer>=2.2.0",
            "presidio-anonymizer>=2.2.0",
            "flair>=0.15.0",
            "spacy>=3.8.0",
            "litserve",
            "pdf2bib",
            "pony",
            "python-dotenv",
        ]
    },
    python_requires = ">=3.9",
)
