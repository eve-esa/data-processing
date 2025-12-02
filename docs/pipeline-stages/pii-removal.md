# PII Removal Stage

This stage detects and redacts **NAMES** and **EMAILS** to protect privacy and ensure compliance. We use the [Presidio](https://github.com/microsoft/presidio) framework with `flair/ner-english-large` model to detect the entities.


### PII Server

You need to setup the PII server found under the `/server`

```bash
cd server
python3 pii_server.py
```

## Configuration

### Basic Configuration

```yaml
- name: pii
  config:
    url: "http://127.0.0.1:8000"
```

## Configuration Parameters

### url
- **Type**: URL
- **Default**: `http://127.0.0.1:8000`
- **Description**: The endpoint for the pii server.

## Next Steps

- Learn about [metadata extraction](metadata-extraction.md)
- Configure [document export](export.md)