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

## Complete End-to-End Pipeline: Process and Upload to Qdrant

This example demonstrates a complete pipeline that extracts, chunks, filters, embeds, and uploads documents to Qdrant in one workflow.

**Prerequisites:**
- VLLM server running for embeddings: `python server/vllm.py`
- Qdrant instance running: `docker run -p 6333:6333 qdrant/qdrant`

```yaml
# examples/process_and_upload.yaml
pipeline:
  batch_size: 10
  inputs:
    path: "data/doc_w_metadata.jsonl"

  stages:
    # Step 1: Extract content from documents
    - name: extraction
      config: { format: "jsonl" }

    # Step 2: Chunk documents into semantic pieces
    - name: chunker
      config: {
        "chunk_overlap": 0,
        "max_chunk_size": 512,
        "word_overlap": 0,
        "add_headers": false,
        "merge_small_chunks": true,
        "headers_to_split_on": [ 1, 2, 3, 4, 5, 6 ]
      }

    # Step 3: Remove short chunks (< 40 words)
    - name: length_filter
      config:
        length: 40
        comparison: "greater"
        action: "keep"

    # Step 4: Remove long chunks (>= 1024 words)
    - name: length_filter
      config:
        length: 1024
        comparison: "less"
        action: "keep"

    # Step 5: Remove references and acknowledgements
    - name: reference_filter
      config:
        action: "discard"

    # Step 6: PII filter with threshold
    - name: pii_filter
      config:
        threshold: 0.03
        action: "discard"
        apply_filter: true

    # Step 7: Remove chunks with excessive newlines
    - name: newline_filter
      config:
        chunks: 60
        comparison: "less"
        action: "keep"

    # Step 8: Generate embeddings and upload to Qdrant
    - name: qdrant_upload
      config:
        mode: "qdrant"
        use_existing_embeddings: false
        upload_pipeline_metadata: true

        embedder:
          model_name: "Qwen/Qwen3-Embedding-4B"
          url: 'http://0.0.0.0:8000'
          timeout: 300
          api_key: "EMPTY"

        vector_store:
          batch_size: 1000
          collection_name: "your-collection-name"
          vector_size: 2560
          url: "http://localhost:6333"
          api_key: "your-api-key"
```

**To run:**
```bash
cp examples/process_and_upload.yaml config.yaml
# Edit config.yaml to set your Qdrant collection name, URL, and API key
eve run
```

**What this pipeline does:**
1. Extracts content from JSONL documents
2. Splits documents into chunks of up to 512 words
3. Filters chunks by length (40-1024 words)
4. Removes references and acknowledgements sections
5. Filters out chunks with PII above 3% threshold
6. Removes chunks with excessive newlines
7. Generates embeddings using VLLM server
8. Uploads filtered documents with embeddings to Qdrant

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