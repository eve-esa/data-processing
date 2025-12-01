# Cleaning Stage

The cleaning stage improves document quality by removing noise artifacts, correcting formatting issues, and enhancing readability.

## Features

- **Noise Removal**: Eliminates OCR related artifacts and noise
- **LaTeX Correction**: Fixes mathematical equations and notation

## Configuration

### Basic Configuration

```yaml
- name: cleaning
  config:
    ocr_threshold: 0.99
```

### LLM-Enhanced Cleaning

You need to set the .env key for `OPENROUTER_API_KEY`.

```yaml
- name: cleaning
  config:
    enable_latex_correction: true
```

## Configuration Parameters

### ocr_threshold
- **Type**: Float
- **Default**: `0.99`
- **Description**: OCR duplicate threshold

### min_words
- **Type**: Int
- **Default**: 2
- **Description**: Minimum words for processing

### enable_latex_correction
- **Type**: Boolean
- **Default**: `false`
- **Description**: Use LLM to fix latex formulas and tables


## Next Steps

- Learn about [PII removal](pii-removal.md)
- Configure [metadata extraction](metadata-extraction.md)
- Set up [document export](export.md)