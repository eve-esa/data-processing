# Configuration Guide

This guide covers the main configuration options for the EVE Pipeline. You can find the detailed configurations under each `Pipeline Stage`.

## Configuration File Structure

The pipeline is configured using a YAML file (typically `config.yaml`) with the following structure:

```yaml
pipeline:
  batch_size: integer
  inputs:
    path: string
    # ... other input options
  stages:
    - name: string
      config: object
    # ... more stages
```

## Global Configuration

### batch_size
- **Type**: Integer
- **Default**: `10`
- **Description**: Number of documents to process in each batch
- **Note**: Not applicable to deduplication stage

```yaml
pipeline:
  batch_size: 20
```

### inputs

#### path
- **Type**: String
- **Required**: Yes
- **Description**: Path to input directory or file containing documents
- **Supported Formats**: Directories with PDF/HTML/XML/Markdown files, or JSONL files

```yaml
pipeline:
  inputs:
    path: "input_documents"  # Directory with various document formats
    # OR
    path: "data/documents.jsonl"  # JSONL file with structured data
```

#### Using JSONL Input Files

JSONL (JSON Lines) format is a powerful way to provide pre-structured documents with metadata. Each line in the file must be a valid JSON object.

**Required Fields:**
- `content` (string): The document text content

**Optional Fields:**
- `metadata` (object): Custom metadata preserved throughout the pipeline
- `embedding` (array): Pre-computed embedding vector
- `pipeline_metadata` (object): Internal metadata from previous pipeline runs

**Example JSONL file (`documents.jsonl`):**
```jsonl
{"content": "First document text here.", "metadata": {"title": "Document 1", "author": "John Doe", "year": 2024}}
{"content": "Second document text.", "metadata": {"title": "Document 2", "source": "research.pdf", "tags": ["AI", "ML"]}}
{"content": "Document with embedding.", "metadata": {"title": "Doc 3"}, "embedding": [0.123, 0.456, 0.789, ...]}
```

**Key Benefits:**
- **Metadata Preservation**: All metadata fields are preserved throughout the pipeline
- **Metadata Inheritance**: When chunking, each chunk inherits the original document's metadata
- **Pre-computed Embeddings**: Can include embeddings to skip re-computation
- **Flexible Schema**: Add any custom metadata fields you need

**Usage Example:**
```yaml
pipeline:
  inputs:
    path: "data/papers.jsonl"
  stages:
    - name: extraction
      config: { format: "jsonl" }
    - name: chunker
      config: { max_chunk_size: 512 }
    # Chunks will have metadata: {"title": "...", "author": "...", "year": ..., "headers": [...]}
```

## Pipeline Stages

### Extraction Stage

Extracts content from various document formats.

```yaml
- name: extraction
  config:
    format: ""  # or "pdf", "html", "xml", "markdown", "jsonl"
    url: "http://127.0.0.1:8001"  # for server-based extraction
```

#### Options

- **format**: Document format specification

    - `""` (default): Automatically detect format
    - `"pdf"`: PDF documents
    - `"html"`: HTML documents
    - `"xml"`: XML documents
    - `"markdown"`: Markdown documents
    - `"jsonl"`: JSON Lines format (one JSON object per line)

- **url**: Server URL for nougat extraction (required for PDF format)

**Note on JSONL Format**: When using JSONL input, each line must be a valid JSON object with a required `content` field. See the [JSONL Input Files](#using-jsonl-input-files) section above for detailed format specifications and examples.

### Deduplication Stage

Removes duplicate and near-duplicate documents.

```yaml
- name: duplication
  config:
    method: "exact"  # or "lsh"
    # LSH options (when method: "lsh")
    shingle_size: 3
    num_perm: 128
    threshold: 0.8
```

#### Options

- **method**: Deduplication method

    - `"exact"` (default): Exact hash-based deduplication
    - `"lsh"`: Locality Sensitive Hashing for near-duplicates

#### LSH Options

- **shingle_size**: Size of text shingles (default: `3`)
- **num_perm**: Number of permutations (default: `128`)
- **threshold**: Similarity threshold (default: `0.8`)

### Cleaning Stage

Removes noise and improves document quality.

```yaml
- name: cleaning
  config:
    ocr_threshold: 0.9
    min_words: 2
    enable_latex_correction: True
```

#### Options

- **ocr_threshold**: OCR duplicate threshold (default: `0.99`)
- **min_words**: Minimum words for processing (default: `2`)
- **enable_latex_correction**: Use LLM to fix latex formulas and tables (default: `false`)


### PII Removal Stage

Redacts personally identifiable information.

```yaml
- name: pii
  config:
    url: "http://127.0.0.1:8000"
```

#### Options

- **url**: Presidio server URL 


### Export Stage

Saves processed documents to output.

```yaml
- name: export
  config:
    format: "md"  # or "txt", "json"
    destination: "output"
```

#### Options

- **format**: Output format

    - `"md"` (default): Markdown
    - `"txt"`: Plain text
    - `"json"`: JSON with metadata

- **destination**: Output directory path