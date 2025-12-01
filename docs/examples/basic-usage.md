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
# pdf_pipeline.yaml
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
# web_content.yaml
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
# scientific_pipeline.yaml
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
        use_llm: true
        correct_latex: true
        correct_tables: true
        llm_url: "http://127.0.0.1:8002"
    - name: pii
      config:
        url: "http://127.0.0.1:8000"
        entities: ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
    - name: metadata
      config:
        extract_pdf_metadata: true
        extract_html_metadata: true
        fields: ["title", "authors", "doi", "year", "journal"]
    - name: export
      config:
        format: "json"
        destination: "final_output"
        include_metadata: true
        filename_pattern: "{doc_id}_{timestamp}"
```

## Batch Processing Examples

### Small Documents (Email, Messages)

```yaml
# small_docs.yaml
pipeline:
  batch_size: 50
  inputs:
    path: "emails"
  stages:
    - name: extraction
    - name: pii
      config:
        url: "http://127.0.0.1:8000"
    - name: export
      config:
        format: "md"
        destination: "cleaned_emails"
```

### Large Documents (Books, Reports)

```yaml
# large_docs.yaml
pipeline:
  batch_size: 2
  inputs:
    path: "technical_reports"
  stages:
    - name: extraction
      config:
        format: "pdf"
    - name: cleaning
      config:
        use_llm: true
    - name: metadata
      config:
        extract_pdf_metadata: true
    - name: export
      config:
        format: "md"
        destination: "processed_reports"
```

## Mixed Format Processing

Process different document types in the same pipeline:

```yaml
# mixed_format.yaml
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
# extract_only.yaml
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
# dedup_only.yaml
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

### Cleaning Only

```yaml
# clean_only.yaml
pipeline:
  batch_size: 15
  inputs:
    path: "noisy_content"
  stages:
    - name: cleaning
      config:
        use_llm: true
    - name: export
      config:
        format: "md"
        destination: "clean_content"
```

## Environment-Specific Configurations

### Development Configuration

```yaml
# dev_config.yaml
pipeline:
  batch_size: 3  # Small batches for debugging
  inputs:
    path: "test_data"
  stages:
    - name: extraction
    - name: cleaning
    - name: export
      config:
        format: "md"
        destination: "dev_output"
```

### Production Configuration

```yaml
# prod_config.yaml
pipeline:
  batch_size: 50  # Large batches for efficiency
  inputs:
    path: "/data/production_documents"
  stages:
    - name: extraction
      config:
        url: "http://extraction-service:8001"
    - name: duplication
      config:
        method: "lsh"
        num_perm: 256
        threshold: 0.85
    - name: cleaning
      config:
        use_llm: true
        llm_url: "http://cleaning-service:8002"
    - name: pii
      config:
        url: "http://pii-service:8000"
    - name: metadata
      config:
        extract_pdf_metadata: true
    - name: export
      config:
        format: "json"
        destination: "/data/processed_documents"
        include_metadata: true
```

## Command Line Usage

### Basic Execution

```bash
# Run with default config file (config.yaml)
eve run

# Run with specific config file
eve run --config custom_config.yaml

# Enable debug logging
eve run --log-level DEBUG

# Run only specific stages
eve run --stages extraction,export
```

### Environment Variables

```bash
# Set batch size
export EVE_BATCH_SIZE=20

# Set input path
export EVE_INPUT_PATH="./my_documents"

# Set export format
export EVE_EXPORT_FORMAT="json"

# Run with environment variables
eve run
```

## Monitoring Progress

### Basic Monitoring

```bash
# The pipeline automatically shows progress:
[2024-01-15 10:30:00] INFO: Processing 1000 files with batch size 10
[2024-01-15 10:30:05] INFO: Processing batch 1/100 (10 documents)
[2024-01-15 10:30:15] INFO: Processing batch 2/100 (20 documents)
[2024-01-15 10:35:00] INFO: Pipeline completed in 300.45 seconds
[2024-01-15 10:35:00] INFO: Processed 850 documents successfully
```

### Custom Logging

```yaml
# config_with_logging.yaml
pipeline:
  batch_size: 10
  inputs:
    path: "documents"
  stages:
    - name: extraction
      config:
        debug: true  # Enable debug for this stage
    - name: export
      config:
        destination: "output"
```

## Error Handling

### Continue on Errors

```yaml
# robust_config.yaml
pipeline:
  batch_size: 10
  inputs:
    path: "documents"
  stages:
    - name: extraction
      config:
        continue_on_error: true
        error_output: "extraction_errors.log"
    - name: export
      config:
        destination: "output"
```

### Validation and Quality Checks

```yaml
# quality_config.yaml
pipeline:
  batch_size: 10
  inputs:
    path: "documents"
  stages:
    - name: extraction
      config:
        validate_content: true
        min_content_length: 100
        max_content_length: 1000000
    - name: export
      config:
        destination: "validated_output"
```

## Next Steps

- Explore [advanced configuration examples](advanced-configuration.md)
- Learn about [server setup](server-setup.md)
- Check the [API reference](../api/) for programmatic usage