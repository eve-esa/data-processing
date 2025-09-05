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
    python_requires = ">=3.9",
)
