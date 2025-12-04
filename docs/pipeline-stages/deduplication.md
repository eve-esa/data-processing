# Deduplication Stage

The deduplication stage removes duplicate and near-duplicate documents from your dataset, improving data quality and reducing processing overhead.

## Deduplication Methods

### Exact Deduplication

Uses SHA-256 checksums to identify identical documents:

```yaml
- name: duplication
  config:
    method: "exact"
```


### LSH (Locality Sensitive Hashing)

Finds near-duplicates using MinHash:

```yaml
- name: duplication
  config:
    method: "lsh"
    shingle_size: 3
    num_perm: 128
    threshold: 0.8
```

## Configuration Parameters

### LSH Parameters

#### shingle_size
- **Type**: Integer
- **Default**: `3`
- **Description**: Size of text chunks (shingles) for comparison. Larger shingles are more specific but increases computation.


#### num_perm
- **Type**: Integer
- **Default**: `128`
- **Description**: Number of random permutations for MinHash. Higher values increase accuracy but use more memory


#### threshold
- **Type**: Float
- **Default**: `0.8`
- **Range**: 0.0-1.0
- **Description**: Similarity threshold for duplicate detection. Higher values find closer duplicates but may miss some.


## Next Steps

- Learn about [content cleaning](cleaning.md)
- Configure [PII removal](pii-removal.md)
- Set up [metadata extraction](metadata-extraction.md)