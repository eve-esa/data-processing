# Chunking

The chunking stage splits large documents into smaller, semantically meaningful chunks that are suitable for downstream processing like embedding generation and vector database upload.

## Overview

Chunking is essential for:

- **Vector database upload**: Breaking documents into appropriately-sized pieces for embedding
- **Semantic retrieval**: Creating chunks that represent coherent topics or concepts
- **Context window management**: Ensuring chunks fit within model token limits
- **Performance optimization**: Parallelizing processing across multiple chunks

The Eve pipeline uses a sophisticated **two-step chunking strategy** that preserves document structure and special content:

1. **Header-based splitting**: First splits documents by Markdown headers to maintain semantic structure
2. **Sentence-based splitting**: If sections exceed the size limit, further splits them by sentences
3. **Smart merging**: Optionally merges small chunks back together when they share compatible heading levels
4. **Content preservation**: Keeps LaTeX formulas, equations, and tables intact as atomic units

## Features

- **Semantic chunking**: Respects document structure by splitting on Markdown headers
- **LaTeX preservation**: Keeps mathematical formulas and equations together
- **Table preservation**: Maintains tables as complete units without splitting
- **Configurable overlap**: Add word-based overlap between chunks for better retrieval
- **Parallel processing**: Uses multiprocessing for fast chunking of large document sets
- **Header inclusion**: Optionally adds section headers to chunks for context

## Configuration

**Step name**: `chunker`

### Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `max_chunk_size` | int | No | `512` | Maximum size of any chunk in words |
| `chunk_overlap` | int | No | `0` | Number of characters to overlap between chunks during secondary splitting |
| `word_overlap` | int | No | `0` | Number of words to overlap between chunks (takes precedence over chunk_overlap) |
| `add_headers` | bool | No | `False` | Whether to prepend section headers to chunk content |
| `merge_small_chunks` | bool | No | `True` | Whether to merge small chunks that share compatible heading levels |
| `headers_to_split_on` | list[int] | No | `[1, 2, 3, 4, 5, 6]` | Markdown header levels to split on (1=`#`, 2=`##`, etc.) |
| `max_workers` | int | No | `None` | Number of parallel workers (None = CPU count) |

### Basic Configuration

```yaml
- name: chunker
  config:
    max_chunk_size: 512
    add_headers: true
    merge_small_chunks: true
```

### Advanced Configuration

```yaml
- name: chunker
  config:
    max_chunk_size: 1024
    chunk_overlap: 0
    word_overlap: 50
    add_headers: true
    merge_small_chunks: true
    headers_to_split_on: [1, 2, 3]  # Only split on H1, H2, H3
    max_workers: 8  # Use 8 parallel workers
```

## How It Works

### Two-Step Chunking Strategy

#### Step 1: Header-Based Splitting

The chunker first splits the document based on Markdown headers:

```markdown
# Introduction
This is the introduction text...

## Background
This is the background section...

## Methods
This section describes methods...
```

This creates initial chunks at natural document boundaries.

#### Step 2: Size-Based Splitting

If any chunk exceeds `max_chunk_size`, it's further split using sentence boundaries while preserving:

- LaTeX environments (`\begin{...}...\end{...}`)
- Inline and display math (`$...$`, `$$...$$`)
- Markdown tables
- Figure and table references

#### Step 3: Smart Merging

If `merge_small_chunks: true`, the chunker merges adjacent chunks when:

1. Combined length doesn't exceed `max_chunk_size`
2. Chunks have compatible heading levels:
   - Same level headers (e.g., two H2 sections)
   - Previous chunk has higher level header (e.g., H1 followed by H2)

This prevents overly small chunks while maintaining semantic coherence.

### Header Inclusion

When `add_headers: true`, section headers are prepended to each chunk:

**Without headers:**
```
This section describes the methodology used in the study...
```

**With headers:**
```
# Introduction
## Methods
This section describes the methodology used in the study...
```

This provides context for each chunk, especially useful for retrieval systems.

### Content Preservation

The chunker intelligently handles special content:

**LaTeX Formulas:**
```latex
The equation \begin{equation}
E = mc^2
\end{equation} is preserved intact.
```

**Tables:**
```markdown
| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
```

These are never split mid-formula or mid-table, even if they exceed `max_chunk_size`.

## Use Cases

### Small Chunks for Dense Retrieval

```yaml
- name: chunker
  config:
    max_chunk_size: 256
    word_overlap: 20
    add_headers: true
    merge_small_chunks: false
```

Creates small, focused chunks with overlap for better semantic retrieval.

### Large Chunks for Context

```yaml
- name: chunker
  config:
    max_chunk_size: 2048
    add_headers: false
    merge_small_chunks: true
    headers_to_split_on: [1, 2]  # Only split on major sections
```

Creates larger chunks that preserve more context, suitable for summarization or large context windows.

### Academic Papers

```yaml
- name: chunker
  config:
    max_chunk_size: 512
    add_headers: true
    merge_small_chunks: true
    headers_to_split_on: [1, 2, 3, 4, 5, 6]
```

Respects the hierarchical structure of academic papers while maintaining readable chunk sizes.

## Output Format

Each chunk becomes a separate `Document` with:

- **content**: The chunk text (with headers if `add_headers: true`)
- **file_path**: Original document file path
- **file_format**: Original document format
- **metadata.headers**: List of Markdown headers that apply to this chunk

### Example Output

```python
Document(
    content="# Introduction\n## Background\nThis paper discusses...",
    file_path="papers/paper1.pdf",
    file_format="pdf",
    metadata={
        "headers": ["#Introduction", "##Background"],
        # ... other metadata from original document
    }
)
```

## Performance

The chunker uses **parallel processing** to handle large document sets efficiently:

- Documents are processed in separate processes using `ProcessPoolExecutor`
- Each process runs an independent chunker instance
- Results are collected and flattened into a single list
- Set `max_workers` to control parallelism (defaults to CPU count)

**Performance tip**: For I/O-bound operations, use the default `max_workers=None`. For CPU-intensive chunking of very large documents, experiment with different worker counts.

## Integration with Other Steps

### Typical Pipeline Order

```yaml
pipeline:
  inputs:
    path: "documents"
  stages:
    - name: extraction

    - name: deduplication
      config:
        method: "lsh"

    - name: cleaning

    - name: chunker
      config:
        max_chunk_size: 512
        add_headers: true

    - name: embedding  # Or qdrant upload

    - name: export
```

### Before Vector Database Upload

Chunking is typically done **before** uploading to vector databases:

```yaml
- name: chunker
  config:
    max_chunk_size: 512
    add_headers: true

- name: qdrant
  config:
    database:
      collection_name: "documents"
    # ... other qdrant config
```

This ensures each chunk gets its own embedding vector in the database.

## Best Practices

1. **Choose appropriate chunk size**:
   1. Smaller chunks (256-512 words) for dense retrieval
   2. Larger chunks (1024-2048 words) for summarization or large context models

2. **Add headers for context**: Enable `add_headers: true` when chunks will be retrieved without surrounding context

3. **Merge small chunks**: Keep `merge_small_chunks: true` to avoid tiny chunks that lack sufficient context

4. **Adjust header levels**: For documents with deep nesting, limit `headers_to_split_on` to major sections only


## Troubleshooting

### Chunks are too large

- Decrease `max_chunk_size`
- Add more header levels to `headers_to_split_on`
- Set `merge_small_chunks: false`

### Chunks are too small

- Increase `max_chunk_size`
- Set `merge_small_chunks: true`
- Reduce header levels in `headers_to_split_on`

### LaTeX formulas are broken

The chunker should preserve LaTeX automatically. If formulas are breaking:

- Check that LaTeX uses proper `\begin{...}` and `\end{...}` syntax
- Verify formulas aren't malformed in the original document
- Review the cleaning step output before chunking

### Slow performance

- Adjust `max_workers` (try different values)
- Ensure you're chunking **after** deduplication and cleaning
- Consider increasing `max_chunk_size` to reduce total chunk count

## Next Steps

- Set up [Qdrant upload](qdrant.md) to store chunks in a vector database
- Learn about [Export options](export.md) for saving chunked documents

## Code Reference

::: eve.steps.chunking.chunker_step
