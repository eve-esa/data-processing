# Extraction Stage

The extraction stage is responsible for reading and extracting content from various document formats. It's the first stage in most pipeline configurations.

## Supported Formats

- **PDF**: Portable Document Format files
- **HTML**: Hypertext Markup Language files
- **XML**: Extensible Markup Language files
- **Markdown**: Markdown text files
- **JSONL**: JSON Lines format (one JSON object per line)

## Configuration

### Basic Configuration

```yaml
- name: extraction
  config:
    format: "pdf"  # or , "html", "xml", "markdown"
```

## Stage Behavior

### Input Processing

The extraction stage processes documents from the configured input directory:

```yaml
pipeline:
  inputs:
    path: "input_documents"
```

- Recursively scans the input directory
- Supports nested folder structures


## Format-Specific Features

### PDF Extraction

For PDF documents, the extractor:

- Extracts text content using [Nougat OCR](https://github.com/facebookresearch/nougat).
- Preserves document structure (headings, paragraphs).
- Maintains table and formulas.

```yaml
- name: extraction
  config:
    format: "pdf"
```

### Nougat Server

You need to setup the nougat server found under the `/server`

```bash
cd server
python3 nougat_server.py
```

### HTML Extraction

For HTML documents, the extractor use [Trafilatura](https://github.com/adbar/trafilatura) to extract the content.

```yaml
- name: extraction
  config:
    format: "html"
```

### XML Extraction

For XML documents, the extractor:

- Extracts text content from XML tags
- Preserves document structure
- Handles namespaces appropriately
- Maintains attribute information when relevant

```yaml
- name: extraction
  config:
    format: "xml"
```

### JSONL Extraction

JSONL (JSON Lines) format allows you to input pre-structured documents with custom metadata. Each line in the file must be a valid JSON object.

**Format Requirements:**

**Required Fields:**
- `content` (string): The document text content

**Optional Fields:**
- `metadata` (object): Custom metadata that will be preserved throughout the pipeline
- `embedding` (array): Pre-computed embedding vector (useful when using `use_existing_embeddings: true`)
- `pipeline_metadata` (object): Internal metadata from previous pipeline runs

**Example JSONL file:**

```jsonl
{"content": "This is the first document.", "metadata": {"title": "Document 1", "author": "John Doe", "year": 2024}}
{"content": "Second document with tags.", "metadata": {"title": "Doc 2", "source": "paper.pdf", "tags": ["AI", "ML"]}}
{"content": "Document with pre-computed embedding.", "metadata": {"title": "Doc 3"}, "embedding": [0.123, 0.456, ...]}
```

**Configuration:**

```yaml
pipeline:
  inputs:
    path: "data/documents.jsonl"
  stages:
    - name: extraction
      config:
        format: "jsonl"
```

**Key Features:**

1. **Flexible Metadata**: Add any custom fields you need (title, author, tags, year, etc.)
2. **Metadata Preservation**: All metadata fields are preserved throughout the entire pipeline
3. **Metadata Inheritance**: When documents are chunked, each chunk inherits the original document's metadata
4. **Pre-computed Embeddings**: Include embeddings to skip re-computation in later stages
5. **Pipeline Chaining**: Output from one pipeline can be input to another via JSONL export

**Practical Example:**

```yaml
pipeline:
  inputs:
    path: "research_papers.jsonl"
  stages:
    - name: extraction
      config: { format: "jsonl" }
    - name: chunker
      config: { max_chunk_size: 512 }
    - name: qdrant_upload
      config:
        mode: "qdrant"
        # ... other config
```

After chunking, each chunk will have metadata like:
```json
{
  "title": "Document 1",
  "author": "John Doe",
  "year": 2024,
  "headers": ["#Introduction", "##Background"]
}
```

This metadata is then uploaded to Qdrant, making it easy to filter and search by author, year, or other custom fields.

## Next Steps

- Learn about [deduplication](deduplication.md)
- Explore [cleaning options](cleaning.md)
- Configure [PII removal](pii-removal.md)