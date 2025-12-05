# Basic Usage Examples

This section provides practical examples of using the EVE Pipeline for common document processing tasks.

## Simple Document Processing

Process all documents from an input directory and export to markdown:

```yaml
# config.yaml
pipeline:
  batch_size: 10
  inputs:
    path: "input_documents"
  stages:
    - name: extraction
    - name: export
      config:
        format: "md"
        destination: "output"
```

```bash
# Run the pipeline
eve run
```

## PDF Processing Pipeline

Process PDF documents with cleaning and deduplication:

```yaml
pipeline:
  batch_size: 5
  inputs:
    path: "research_papers"
  stages:
    - name: extraction
      config:
        format: "pdf"
    - name: duplication
    - name: cleaning
    - name: export
      config:
        format: "md"
        destination: "processed_papers"
```

## Web Content Processing

Process HTML documents with PII removal:

```yaml
pipeline:
  batch_size: 20
  inputs:
    path: "web_pages"
  stages:
    - name: extraction
      config:
        format: "html"
    - name: pii
      config:
        url: "http://127.0.0.1:8000"
    - name: export
      config:
        format: "txt"
        destination: "clean_content"
```

## Advanced Pipeline with All Features

Complete pipeline for scientific document processing:

```yaml
pipeline:
  batch_size: 10
  inputs:
    path: "scientific_documents"
  stages:
    - name: extraction
      config:
        url: "http://127.0.0.1:8001"
    - name: duplication
      config:
        method: "lsh"
        shingle_size: 3
        num_perm: 128
        threshold: 0.85
    - name: cleaning
      config:
        ocr_threshold: 0.99
        enable_latex_correction: true
        debug: true
    - name: pii
      config:
        url: "http://127.0.0.1:8000"
    - name: metadata
      config:
    - name: export
      config:
        export_metadata: true
        metadata_destination: "./output"
```


## Mixed Format Processing

Process different document types in the same pipeline:

```yaml
pipeline:
  batch_size: 10
  inputs:
    path: "mixed_documents"
  stages:
    - name: extraction
      # Auto-detect format based on file extension
    - name: duplication
      config:
        method: "lsh"
        threshold: 0.8
    - name: cleaning
    - name: export
      config:
        format: "md"
        destination: "unified_output"
```

## Selective Stage Processing

Skip certain stages based on your needs:

### Extraction Only

```yaml
pipeline:
  batch_size: 20
  inputs:
    path: "raw_documents"
  stages:
    - name: extraction
    - name: export
      config:
        format: "md"
        destination: "extracted_content"
```

### Deduplication Only

```yaml
pipeline:
  inputs:
    path: "markdown_documents"
  stages:
    - name: duplication
      config:
        method: "lsh"
        threshold: 0.9
    - name: export
      config:
        format: "md"
        destination: "unique_documents"
```