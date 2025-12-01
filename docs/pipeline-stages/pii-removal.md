# PII Removal Stage

The PII (Personally Identifiable Information) removal stage automatically detects and redacts **NAMES** and **EMAILS** to protect privacy and ensure compliance.


## Configuration

### Basic Configuration

```yaml
- name: pii
  config:
    url: "http://127.0.0.1:8000"
```


## Next Steps

- Learn about [metadata extraction](metadata-extraction.md)
- Configure [document export](export.md)