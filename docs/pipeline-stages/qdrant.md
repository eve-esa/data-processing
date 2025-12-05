# Qdrant Upload Stage

The Qdrant upload stage generates embeddings for documents and optionally uploads them to a Qdrant vector database for semantic search and retrieval.

## Features

- **Dual Mode Operation**: Upload to Qdrant database or store embeddings locally
- **Flexible Embedding Sources**: Generate new embeddings or use existing ones from metadata
- **Automatic Deduplication**: Skips documents already present in the collection
- **Batch Processing**: Efficient batch upload with configurable size
- **Optimized Collection Setup**: Automatic creation with HNSW indexing and quantization
- **Metadata Indexing**: Creates indexes on key fields for efficient filtering
- **Retry Logic**: Automatic retry on failures with configurable attempts

## Operating Modes

### Qdrant Mode (Default)

Uploads document embeddings to a Qdrant vector database:

```yaml
- name: qdrant
  config:
    mode: "qdrant"
    vector_store:
      url: "http://localhost:6333"
      api_key: "your-api-key"  # Optional
      collection_name: "my_documents"
      batch_size: 100
      vector_size: 768
    embedder:
      url: "http://localhost:8000"
      model_name: "BAAI/bge-base-en-v1.5"
      timeout: 300
      api_key: "EMPTY"  # Optional
```

### Local Mode

Generates embeddings and stores them in document metadata without uploading:

```yaml
- name: qdrant
  config:
    mode: "local"
    batch_size: 10
    embedder:
      url: "http://localhost:8000"
      model_name: "BAAI/bge-base-en-v1.5"
      timeout: 300
```

## Configuration Parameters

### Mode Configuration

#### mode
- **Type**: String
- **Default**: `"qdrant"`
- **Options**: `"qdrant"`, `"local"`
- **Description**: Operating mode - upload to Qdrant database or store embeddings locally

#### use_existing_embeddings
- **Type**: Boolean
- **Default**: `false`
- **Description**: Use embeddings already stored in `document.embedding` field instead of generating new ones

#### upload_pipeline_metadata
- **Type**: Boolean
- **Default**: `false`
- **Description**: Include pipeline processing metadata in the Qdrant payload

#### batch_size
- **Type**: Integer
- **Default**: `10` (local mode)
- **Description**: Number of documents to process in each batch (for local mode only)

### Vector Store Configuration (Qdrant Mode)

#### vector_store.url
- **Type**: String
- **Default**: `"http://localhost:6333"`
- **Description**: URL of the Qdrant instance

#### vector_store.api_key
- **Type**: String
- **Default**: None
- **Description**: API key for Qdrant authentication (optional for local instances)

#### vector_store.collection_name
- **Type**: String
- **Required**: Yes (for Qdrant mode)
- **Description**: Name of the target collection in Qdrant

#### vector_store.batch_size
- **Type**: Integer
- **Required**: Yes (for Qdrant mode)
- **Description**: Number of documents to upload per batch

#### vector_store.vector_size
- **Type**: Integer
- **Required**: Yes (for Qdrant mode)
- **Description**: Dimension of embedding vectors (must match model output)

### Embedder Configuration

#### embedder.url
- **Type**: String
- **Required**: Yes (unless `use_existing_embeddings: true`)
- **Description**: URL of the VLLM embedding server

#### embedder.model_name
- **Type**: String
- **Required**: Yes (unless `use_existing_embeddings: true`)
- **Description**: Name of the embedding model

#### embedder.timeout
- **Type**: Integer
- **Default**: `300`
- **Description**: Request timeout in seconds

#### embedder.api_key
- **Type**: String
- **Default**: `"EMPTY"`
- **Description**: API key for VLLM authentication (use "EMPTY" for local servers)

## Collection Optimization

When creating a new collection, the step automatically applies optimized settings for production use. These settings balance search quality, speed, and resource usage for large-scale document collections.

[Learn more about Qdrant Collections →](https://qdrant.tech/documentation/concepts/collections/)

<details>
<summary><h3>Vector Configuration</h3></summary>

#### Distance Metric: COSINE
Cosine similarity measures the angle between vectors, making it ideal for text embeddings where the direction matters more than magnitude. This is the standard choice for semantic search applications.

[Read about Distance Metrics →](https://qdrant.tech/documentation/concepts/search/#metrics)

#### On-Disk Storage: Enabled
Stores vectors on disk rather than in RAM, significantly reducing memory requirements for large collections. While slightly slower than in-memory storage, this allows you to store millions of vectors affordably.

#### Shards: 8
Distributes data across 8 shards for parallel processing. More shards improve write throughput and allow better resource utilization, especially important for large datasets.

[Learn about Sharding →](https://qdrant.tech/documentation/guides/distributed_deployment/)

</details>

### HNSW Index Parameters

HNSW (Hierarchical Navigable Small World) is the indexing algorithm that enables fast approximate nearest neighbor search.

[Deep dive into HNSW →](https://qdrant.tech/documentation/concepts/indexing/#vector-index)

#### m: 16
Number of bidirectional links created for each node. Higher values improve search quality but increase memory usage and indexing time. 16 is a balanced choice for most applications.
- **Lower (4-8)**: Less memory, faster indexing, slightly lower recall
- **Higher (32-64)**: Better recall, more memory, slower indexing

#### ef_construct: 128
Size of the dynamic candidate list during index construction. Higher values produce better index quality but take longer to build. 128 provides good quality without excessive build time.
- **Lower (64)**: Faster indexing, slightly lower search quality
- **Higher (256-512)**: Better search quality, slower indexing

#### full_scan_threshold: 10,000
When collection size is below this threshold, Qdrant uses exact (brute-force) search instead of the HNSW index. Exact search is faster for small collections.

#### max_indexing_threads: 2
Limits CPU cores used during indexing. Prevents indexing from consuming all available resources.

#### on_disk: Enabled
Stores the HNSW graph on disk to reduce RAM usage. Essential for collections with millions of vectors.

### Quantization

Binary quantization compresses vectors from 32-bit floats to 1-bit representations, reducing memory by ~32x with minimal quality loss. This makes it possible to store much larger collections.

[Learn about Quantization →](https://qdrant.tech/documentation/guides/quantization/)

#### Type: Binary Quantization
Converts vector components to binary (0 or 1) for massive memory savings. The original vectors are still used for final re-ranking, so search quality remains high.

**Benefits:**
- 32x memory reduction (32-bit float → 1-bit)
- Faster distance calculations
- More vectors fit in RAM for better performance
- Negligible impact on search quality (typically <2% recall loss)

#### always_ram: false
Allows quantized vectors to be stored on disk when needed, rather than always keeping them in RAM. This provides flexibility for very large collections.

### Optimizer Settings

These settings control how Qdrant manages and optimizes data segments over time.

[Learn about Storage Optimization →](https://qdrant.tech/documentation/concepts/storage/)

#### indexing_threshold: 20,000
Build HNSW index when segment reaches this size. Smaller values create indexes sooner but may cause more frequent rebuilds.

#### memmap_threshold: 5,000
Use memory-mapped files for segments larger than this. Memory mapping allows efficient disk-based storage without loading everything into RAM.

#### max_segment_size: 5,000,000
Maximum vectors per segment. Larger segments are more memory-efficient but may slow down some operations.

#### max_optimization_threads: 2
CPU cores dedicated to background optimization tasks. Prevents optimization from impacting query performance.

### Payload Indexes

Payload indexes enable fast filtering on metadata fields, similar to database indexes. Without these, filtering requires scanning all documents.

[Learn about Payload Indexes →](https://qdrant.tech/documentation/concepts/indexing/#payload-index)

The pipeline automatically creates indexes on common academic metadata fields:

#### Text Indexes (title, journal)
Enable full-text search and filtering on text fields. The word tokenizer splits text into searchable terms.
- **min_token_len**: Minimum word length to index
- **max_token_len**: Maximum word length to index
- **lowercase**: Normalize to lowercase for case-insensitive search

**Example filters:**
- Find papers with "neural" in title
- Filter by journal name
- Combine with vector search for semantic + keyword search

#### Integer Indexes (year, n_citations)
Enable efficient range queries on numeric fields.

**Example filters:**
- Papers published after 2020
- Papers with >100 citations
- Combine filters: papers from 2015-2023 with >50 citations

**Performance Impact:**
- Indexes speed up filtering by 100-1000x
- Small storage overhead (~10-20% of original data)
- Slightly slower writes (indexes must be updated)

## Stage Behavior

### Metadata Handling

Document metadata is prepared for storage:
- **content**: Document text content
- **metadata**: All user metadata fields (unwrapped at root level)
- **pipeline_metadata**: Processing metadata (if `upload_pipeline_metadata: true`)

Type conversions:
- `year` field converted to integer
- `title` field converted to string

### Error Handling

- Batch uploads retry up to 3 times on failure
- Failed batches are logged and skipped
- Individual embedding failures are logged without stopping the pipeline
- Scroll operations retry with exponential backoff

## Usage Examples

### Basic Upload with New Embeddings

```yaml
- name: qdrant
  config:
    mode: "qdrant"
    vector_store:
      url: "http://localhost:6333"
      collection_name: "research_papers"
      batch_size: 100
      vector_size: 768
    embedder:
      url: "http://localhost:8000"
      model_name: "BAAI/bge-base-en-v1.5"
```

### Upload with Existing Embeddings

If embeddings were generated in a previous step:

```yaml
- name: qdrant
  config:
    mode: "qdrant"
    use_existing_embeddings: true
    upload_pipeline_metadata: true
    vector_store:
      url: "http://localhost:6333"
      api_key: "your-api-key"
      collection_name: "processed_docs"
      batch_size: 50
      vector_size: 1024
```

### Local Embedding Generation

Store embeddings in document metadata without uploading:

```yaml
- name: qdrant
  config:
    mode: "local"
    batch_size: 20
    embedder:
      url: "http://localhost:8000"
      model_name: "sentence-transformers/all-MiniLM-L6-v2"
      timeout: 600
```

## VLLM Server Setup

The Qdrant step requires a VLLM server for embedding generation. Start the server:

```bash
cd server
python vllm.py
```

The server provides an OpenAI-compatible embeddings API at `/v1/embeddings`.

## Complete Pipeline Example

```yaml
pipeline:
  inputs:
    path: "processed_documents"

  stages:
    - name: chunking
      config:
        max_chunk_size: 512
        chunk_overlap: 50

    - name: metadata
      config:
        enabled_formats: ["pdf", "markdown"]
        enable_scholar_search: true

    - name: qdrant
      config:
        mode: "qdrant"
        upload_pipeline_metadata: true
        vector_store:
          url: "http://localhost:6333"
          collection_name: "academic_papers"
          batch_size: 100
          vector_size: 768
        embedder:
          url: "http://localhost:8000"
          model_name: "BAAI/bge-base-en-v1.5"
          timeout: 300
```

## Next Steps

- Learn about [chunking strategy](chunking.md)
- Configure [metadata extraction](metadata-extraction.md)
- Explore [export options](export.md)

## Code Reference

::: eve.steps.qdrant.qdrant_step