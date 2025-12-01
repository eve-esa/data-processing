# Quick Start

This tutorial will walk you through running your first data processing pipeline with EVE.

## Step 1: Prepare Your Data

Create an input directory with your documents:

```bash
mkdir -p input_data
# Copy your PDF, HTML, XML, or Markdown files here
cp /path/to/your/documents/* input_data/
```

## Step 2: Basic Configuration

Create a `config.yaml` file:

```yaml
pipeline:
  batch_size: 10
  inputs:
    path: "input_data"
  stages:
    - name: extraction
      # Automatically detects file format
    - name: duplication
    - name: export
      config:
        format: "md"
        destination: "output"
```

## Step 3: Run the Pipeline

Execute the pipeline:

```bash
eve run
```

## Step 4: Check Results

Your processed documents will be in the `output` directory:

```bash
ls output/
```

## Example Pipeline Configurations

### PDF Processing Only

```yaml
pipeline:
  batch_size: 5
  inputs:
    path: "pdfs"
  stages:
    - name: extraction
      config: { format: "pdf" }
    - name: cleaning
    - name: export
      config: { format: "md", destination: "processed_pdfs" }
```

### HTML Processing with PII Removal

```yaml
pipeline:
  batch_size: 10
  inputs:
    path: "html_docs"
  stages:
    - name: extraction
      config: { format: "html", url: "http://127.0.0.1:8001" }
    - name: pii
      config: { url: "http://127.0.0.1:8000" }
    - name: export
      config: { format: "md"}
```

### Advanced Pipeline with All Stages

```yaml
pipeline:
  batch_size: 10
  inputs:
    path: "mixed_docs"
  stages:
    - name: extraction
      config: { url: "http://127.0.0.1:8001" }
    - name: duplication
      config: {
        method: "lsh",
        shingle_size: 3,
        num_perm: 128,
        threshold: 0.8
      }
    - name: cleaning
    - name: pii
      config: { url: "http://127.0.0.1:8000" }
    - name: metadata
```

## Monitoring Progress

The pipeline provides progress updates:

```bash
$ eve run

[2024-01-15 10:30:00] INFO: Starting pipeline with 100 documents
[2024-01-15 10:30:01] INFO: Stage 1/5: Extraction
[2024-01-15 10:30:15] INFO: Processing batch 1/10 (10 documents)
[2024-01-15 10:30:30] INFO: Processing batch 2/10 (20 documents)
...
[2024-01-15 10:35:00] INFO: Pipeline completed successfully
[2024-01-15 10:35:00] INFO: Processed 95 documents, 5 duplicates removed
```

## Next Steps

- Learn about [configuration options](configuration.md)
- Explore [pipeline stages](../pipeline-stages/)
- Check out [advanced examples](../examples/)