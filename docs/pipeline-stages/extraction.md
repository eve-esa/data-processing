# Extraction Stage

The extraction stage is responsible for reading and extracting content from various document formats. It's the first stage in most pipeline configurations.

## Supported Formats

- **PDF**: Portable Document Format files
- **HTML**: Hypertext Markup Language files
- **XML**: Extensible Markup Language files
- **Markdown**: Markdown text files

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

- Extracts text content using Nougat OCR.
- Preserves document structure (headings, paragraphs).
- Maintains table and formulas.

```yaml
- name: extraction
  config:
    format: "pdf"
```

### HTML Extraction

For HTML documents, the extractor use trafilatura to extract the content.

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

## Next Steps

- Learn about [deduplication](deduplication.md)
- Explore [cleaning options](cleaning.md)
- Configure [PII removal](pii-removal.md)