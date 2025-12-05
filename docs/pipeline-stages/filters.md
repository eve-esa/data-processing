# Filters

Filters allow you to selectively keep or discard documents based on specific criteria. They are essential for quality control and ensuring your processed documents meet your requirements.

## Overview

Filters in the Eve pipeline evaluate documents against configurable criteria and either keep or discard them. Each filter:

- Adds metadata to documents (e.g., word count, PII percentage)
- Can be configured with `keep` or `discard` actions
- Provides detailed logging of filtering results
- Can be chained together for complex filtering logic

## Available Filters

### Length Filter

Filters documents based on word count, useful for removing documents that are too short or too long.

**Step name**: `length_filter`

#### Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `length` | int | Yes | - | Word count threshold for filtering |
| `comparison` | str | No | `"greater"` | Either `"less"` or `"greater"` to compare against threshold |
| `action` | str | No | `"keep"` | Either `"keep"` or `"discard"` documents matching the condition |

#### Examples

```yaml
# Keep only documents with more than 1000 words
- name: length_filter
  config:
    length: 1000
    comparison: "greater"
    action: "keep"
```

```yaml
# Remove short documents (less than 100 words)
- name: length_filter
  config:
    length: 100
    comparison: "less"
    action: "discard"
```

```yaml
# Create a range filter: keep documents between 50-1000 words
# This requires chaining two length filters
- name: length_filter
  config:
    length: 50
    comparison: "greater"
    action: "keep"

- name: length_filter
  config:
    length: 1000
    comparison: "less"
    action: "keep"
```

#### Metadata Added

- `word_count`: Number of words in the document

---

### PII Filter

Filters documents based on the percentage of Personally Identifiable Information (PII) tokens. Useful for removing documents with excessive PII or keeping only anonymized documents.

**Step name**: `pii_filter`

**Note**: This filter expects PII tokens to already be marked in the document (e.g., `[PERSON]`, `[EMAIL_ADDRESS]`). Use the [PII removal step](pii-removal.md) before this filter to anonymize PII.

#### Special Behavior

Documents containing "abstract" or "introduction" sections are **always kept** regardless of PII percentage. This is because academic papers often mention author names in these sections.

#### Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `threshold` | float | Yes | - | PII token percentage threshold (e.g., 0.03 for 3%) |
| `action` | str | No | `"discard"` | Either `"keep"` or `"discard"` documents meeting the threshold |
| `apply_filter` | bool | No | `true` | Whether to apply filtering or just calculate PII percentage |

#### Examples

```yaml
# Remove documents with 3% or more PII tokens (keep abstracts/intros)
- name: pii_filter
  config:
    threshold: 0.03
    action: "discard"
```

```yaml
# Keep only documents with low PII (less than 1%)
- name: pii_filter
  config:
    threshold: 0.01
    action: "discard"
```

```yaml
# Just calculate PII percentage without filtering
- name: pii_filter
  config:
    threshold: 0.03
    apply_filter: false
```

#### Metadata Added

- `pii_tokens_percentage`: Percentage of PII tokens in the document (0.0 to 1.0)

---

### Newline Filter

Filters documents based on the number of newline characters, useful for identifying documents with specific formatting characteristics.

**Step name**: `newline_filter`

#### Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `chunks` | int | Yes | - | Newline count threshold for filtering |
| `comparison` | str | No | `"greater"` | Either `"less"` or `"greater"` to compare against threshold |
| `action` | str | No | `"keep"` | Either `"keep"` or `"discard"` documents matching the condition |

#### Examples

```yaml
# Keep documents with more than 10 newlines (well-structured content)
- name: newline_filter
  config:
    chunks: 10
    comparison: "greater"
    action: "keep"
```

```yaml
# Remove documents with too many newlines (likely poorly formatted)
- name: newline_filter
  config:
    chunks: 100
    comparison: "greater"
    action: "discard"
```

#### Metadata Added

- `newline_count`: Number of newline characters in the document

---

### Reference Filter

Filters documents that contain references or acknowledgements sections, useful for academic paper processing.

**Step name**: `reference_filter`

The filter detects references and acknowledgements by checking:

- Document headers metadata for keywords: "reference", "references", "acknowledgement", "acknowledgements"
- Text content for markdown headers containing these keywords

#### Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `action` | str | No | `"discard"` | Either `"keep"` or `"discard"` documents with references/acknowledgements |

#### Examples

```yaml
# Remove documents with references or acknowledgements
- name: reference_filter
  config:
    action: "discard"
```

```yaml
# Keep only documents with references (academic papers)
- name: reference_filter
  config:
    action: "keep"
```

---

### Perplexity Filter

Filters documents based on perplexity scores calculated by a language model. Lower perplexity indicates more natural, coherent text.

**Step name**: `perplexity_filter`

**Note**: This filter loads a language model which requires significant memory and compute resources.

#### Configuration Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `threshold` | float | No | `0.0` | Perplexity threshold for filtering |
| `enable_threshold` | bool | No | `false` | Whether to apply threshold-based filtering |
| `model_name` | str | No | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` | Hugging Face model to use for perplexity calculation |
| `stride` | int | No | `128` | Stride for sliding window perplexity calculation |
| `batch_size` | int | No | `128` | Batch size for model inference |
| `max_length` | int | No | `1024` | Maximum sequence length for model |

#### Examples

```yaml
# Calculate perplexity for all documents without filtering
- name: perplexity_filter
  config:
    model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    enable_threshold: false
```

```yaml
# Keep only documents with perplexity below 50
- name: perplexity_filter
  config:
    threshold: 50.0
    enable_threshold: true
    model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

#### Metadata Added

- `perplexity`: Perplexity score calculated by the language model

---

## Chaining Filters

Filters can be chained to create complex filtering logic. Filters are applied sequentially, with each filter receiving the output of the previous filter.

### Example: Multi-Stage Quality Filter

```yaml
pipeline:
  inputs:
    path: "input_docs"
  stages:
    - name: extraction

    - name: cleaning

    # Remove very short documents
    - name: length_filter
      config:
        length: 50
        comparison: "less"
        action: "discard"

    # Remove very long documents
    - name: length_filter
      config:
        length: 10000
        comparison: "greater"
        action: "discard"

    # Remove documents with reference sections
    - name: reference_filter
      config:
        action: "discard"

    # Remove documents with high PII
    - name: pii_filter
      config:
        threshold: 0.05
        action: "discard"

    - name: export
      config:
        output_dir: "filtered_output"
```

## Best Practices

1. **Order matters**: Apply computationally expensive filters (like perplexity) after cheaper filters (like length) to reduce processing time
2. **Test thresholds**: Start with permissive thresholds and adjust based on your data
3. **Use metadata**: Even if `apply_filter: false`, the metadata added by filters can be useful for analysis
4. **Monitor filtering**: Check the logs to ensure you're not filtering out too many documents
5. **Chain wisely**: Use multiple filters to create precise selection criteria

## Next Steps

- Learn about [Chunking](chunking.md) for splitting documents
- Configure [Metadata Extraction](metadata-extraction.md)
- Set up [Document Export](export.md)

## Code Reference

::: eve.steps.filters.length_filter
::: eve.steps.filters.pii_filter
::: eve.steps.filters.newline_filter
::: eve.steps.filters.reference_filter
::: eve.steps.filters.perplexity
