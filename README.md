# Eve Data Processing Pipeline

A scalable, modular, and production-ready data processing pipeline for document extraction, cleaning, PII removal, and deduplication.

## 🚀 Features

- **Multi-format Extraction**: PDF, XML, HTML, CSV, TXT support
- **5-Stage Cleaning Process**: OCR corrections, deduplication, Nougat postprocessing, rule-based cleaning, artifact removal
- **PII Removal**: Presidio + Flair integration for sensitive data detection
- **Two-Level Deduplication**: Exact hash-based and LSH approximate duplicate detection
- **Modular Architecture**: Independent, configurable pipeline stages
- **High Performance**: Parallel processing with configurable workers
- **Production Ready**: Type hints, comprehensive testing, logging, monitoring

## 📦 Installation

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the pipeline
git clone <repository-url>
cd eve-pipeline
uv sync
```

### Development Installation

```bash
uv sync --dev  # Installs dev dependencies including ruff, pytest, mypy
```

## 🏃 Quick Start

### Command Line Interface

```bash
# Process a single file
eve-pipeline process input.pdf --output output.md

# Process a directory
eve-pipeline process data/ --output processed/ --num-processes 4

# Generate configuration file
eve-pipeline init-config --output my_config.yaml

# Deduplication analysis
eve-pipeline deduplicate data/ --save-duplicates duplicates.json
```

### Python API

```python
from eve_pipeline import Pipeline, PipelineConfig

# Create pipeline with default configuration
pipeline = Pipeline()

# Process a single file
result = pipeline.process_file("document.pdf", "output.md")
if result.is_success:
    print(f"Processed successfully")

# Process a directory
results = pipeline.process_directory("data/", "processed/")
print(f"Processed {results['statistics']['successful']} files")
```

## 🏗️ Architecture

The pipeline consists of 5 main stages:

1. **Extraction**: Multi-format document processing
2. **Cleaning**: 5-check cleaning process
3. **PII Removal**: Sensitive data anonymization  
4. **Deduplication**: Exact and approximate duplicate detection
5. **LaTeX Correction**: Mathematical notation fixes using GPT-4o-mini