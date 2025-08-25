# Eve Data Processing Pipeline

A comprehensive, scalable, and production-ready data processing pipeline for document extraction, cleaning, PII removal, deduplication, and LaTeX correction with async support and rate limiting.

## 🚀 Features

- **Multi-format Extraction**: PDF, XML, HTML, CSV, TXT support with async processing
- **5-Stage Cleaning Process**: OCR corrections, deduplication, Nougat postprocessing, rule-based cleaning, artifact removal
- **PII Removal**: Presidio + Flair integration for sensitive data detection and anonymization
- **Two-Level Deduplication**: Thread-safe exact hash-based and LSH approximate duplicate detection
- **LaTeX Correction**: AI-powered formula validation and correction with OpenRouter GPT-4o-mini
- **Async Support**: Non-blocking storage operations and parallel processing
- **Rate Limiting**: Intelligent OpenRouter API rate limiting with exponential backoff
- **Production Ready**: Type hints, comprehensive configuration validation, centralized logging, monitoring
- **Thread-Safe**: Race condition protection for multi-threaded operations
- **Storage Flexibility**: Support for local and S3 storage with seamless switching

## 📦 Installation

### Prerequisites

- Python 3.9+ (tested on Python 3.9, 3.10, 3.11)
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux | ✅ Fully Supported | Primary development platform |
| macOS | ✅ Fully Supported | Intel and Apple Silicon |
| Windows | ⚠️ Limited Support | See [Windows Setup](#windows-setup) |

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd eve-pipeline

# Install the pipeline with all dependencies
uv sync

# For development installation (includes testing tools)
uv sync --dev
```

### Using pip (Alternative)

```bash
git clone <repository-url>
cd eve-pipeline
pip install -e .

# For development
pip install -e ".[dev]"
```

### Windows Setup

For Windows users, additional steps may be required:

```powershell
# Install uv using PowerShell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv

# Continue with standard installation
uv sync
```

**Note**: Some features (like PDF extraction with Nougat) may require additional system dependencies on Windows.

### Troubleshooting Installation

If you encounter `ModuleNotFoundError: No module named 'pip'`:
```bash
uv pip install pip
```

## 🔧 Setup and Configuration

### Environment Variables (.env)

Create a `.env` file in the project root with the following variables:

```bash
# OpenRouter API Configuration (required for LaTeX correction)
# Uses OpenRouter to access gpt-4o-mini model
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Storage Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
AWS_SESSION_TOKEN=optional_session_token


# Nougat Model Configuration (optional)
NOUGAT_CHECKPOINT=/path/to/nougat/checkpoint

# Logging Configuration
LOG_LEVEL=INFO
```

### Storage Configuration

The pipeline supports multiple storage backends:

#### Local Storage
```yaml
storage:
  save_to_local: true
  local_base_dir: "/path/to/local/storage"
```

#### S3 Storage
```yaml
storage:
  save_to_s3: true
  s3_bucket: "your-bucket-name"
  s3_prefix: "eve-pipeline/"
  aws_region: "us-east-1"
```

#### Mixed Storage (Local + S3)
```yaml
storage:
  save_to_local: true
  save_to_s3: true
  local_base_dir: "/local/path"
  s3_bucket: "backup-bucket"
```

### PII Removal Server Setup

For PII removal functionality, start the PII server:

```bash
# Navigate to PII removal module
cd eve_pipeline/pii_removal/

# Start server with desired number of workers
python server.py --workers 4 --port 8000

# Or use the CLI command
eve-pipeline serve --pii-only --workers 4 --port 8000
```

## 🏃 Quick Start

### Command Line Interface

```bash
# Generate a configuration file
eve-pipeline init-config --output my_config.yaml

# Process a single file
eve-pipeline process document.pdf --output output.md

# Process a directory with custom configuration
eve-pipeline process data/ --output processed/ --config my_config.yaml --num-processes 4

# Process with specific stages enabled/disabled
eve-pipeline process data/ --output processed/ \
  --extraction --cleaning --pii-removal \
  --no-deduplication --latex-correction

# Deduplication analysis only
eve-pipeline deduplicate data/ --save-duplicates duplicates.json

# Process S3 data
eve-pipeline process s3://my-bucket/input/ --output s3://my-bucket/output/
```

### Python API

```python
import asyncio
from eve_pipeline import Pipeline, PipelineConfig

# Create pipeline with default configuration
pipeline = Pipeline()

# Process a single file
result = pipeline.process_file("document.pdf", "output.md")
if result.is_success:
    print(f"Processed successfully in {result.processing_time:.2f}s")
    print(f"Stages completed: {result.metadata.get('stages_completed', 0)}")

# Process a directory
results = pipeline.process_directory("data/", "processed/")
print(f"Processed {results['statistics']['successful']} files")

# Async processing (non-blocking)
async def process_async():
    result = await pipeline.process_file_async("document.pdf", "output.md")
    return result

# Run async processing
result = asyncio.run(process_async())
```

### Advanced Configuration

```python
from eve_pipeline import PipelineConfig, ExtractionMethod, HashAlgorithm, LogLevel

# Create custom configuration
config = PipelineConfig(
    num_processes=8,
    debug=True,
    log_level=LogLevel.DEBUG,
    retry_failed_files=True,
    max_retries=3,
    
    extraction=ExtractionConfig(
        pdf_extractor=ExtractionMethod.NOUGAT,
        batch_size=4,
    ),
    
    deduplication=DeduplicationConfig(
        hash_algorithm=HashAlgorithm.SHA256,
        lsh_threshold=0.85,
    ),
    
    storage=StorageConfig(
        save_to_s3=True,
        s3_bucket="my-pipeline-bucket",
        aws_region="us-west-2",
    )
)

# Initialize pipeline with custom config
pipeline = Pipeline(config=config)
```

## Stage Details

1. **Extraction**: Multi-format document processing (PDF via Nougat/Marker/PyPDF, XML, HTML, CSV, TXT)
2. **Cleaning**: 5-check cleaning process with OCR corrections and artifact removal
3. **PII Removal**: Sensitive data anonymization using Presidio + Flair
4. **Deduplication**: Two-level exact and approximate duplicate detection
5. **LaTeX Correction**: Mathematical notation validation and AI-powered fixes

## 🔧 Configuration Reference

### Pipeline Configuration (`PipelineConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_processes` | int | CPU count | Number of parallel processes |
| `debug` | bool | False | Enable debug logging |
| `log_level` | LogLevel | INFO | Logging verbosity |
| `retry_failed_files` | bool | False | Enable retry mechanism |
| `max_retries` | int | 3 | Maximum retry attempts |

### Extraction Configuration (`ExtractionConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | True | Enable extraction stage |
| `pdf_extractor` | ExtractionMethod | NOUGAT | PDF extraction method |
| `batch_size` | int | 4 | Processing batch size |
| `supported_formats` | List[str] | See code | Supported file extensions |

### Cleaning Configuration (`CleaningConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | True | Enable cleaning stage |
| `ocr_corrections` | bool | True | Fix OCR-induced errors |
| `latex_correction` | bool | True | AI-powered LaTeX correction |
| `openai_api_key` | str | None | OpenRouter/OpenAI API key |

### Deduplication Configuration (`DeduplicationConfig`)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | True | Enable deduplication |
| `exact_deduplication` | bool | True | Enable exact matching |
| `lsh_deduplication` | bool | True | Enable approximate matching |
| `hash_algorithm` | HashAlgorithm | MD5 | Hash algorithm for exact dedup |
| `lsh_threshold` | float | 0.8 | Similarity threshold (0.0-1.0) |

## 🚦 Rate Limiting and Performance

### OpenRouter API Rate Limiting

The pipeline includes intelligent rate limiting for OpenRouter API calls:

- **Requests per minute**: 500 (configurable)
- **Tokens per minute**: 200,000 (configurable)
- **Exponential backoff**: Automatic retry with increasing delays
- **Jitter**: Random delays to prevent thundering herd

### Performance Optimization

- **Async I/O**: Non-blocking file operations
- **Batch Processing**: Memory-efficient processing for large datasets
- **Thread Safety**: Race condition protection
- **Generator-based Processing**: Memory-efficient directory traversal

## 🧪 Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=eve_pipeline

# Run specific test category
uv run pytest tests/test_extraction/
uv run pytest tests/test_deduplication/
```

### Mock Testing

The pipeline includes comprehensive mocking for external dependencies:

```python
# Example test with mocking
import pytest
from unittest.mock import Mock, patch
from eve_pipeline import Pipeline

@patch('eve_pipeline.storage.s3.boto3')
def test_s3_processing(mock_boto3):
    mock_s3 = Mock()
    mock_boto3.client.return_value = mock_s3
    
    pipeline = Pipeline()
    result = pipeline.process_file("s3://bucket/file.pdf")
    
    assert result.is_success
```