# Cleaning Stage

The cleaning stage improves document quality by removing OCR errors, noise artifacts, correcting formatting issues, and enhancing readability.

## Features

- **Noise Removal**: Fixes the errors introduced during the OCR extraction.
- **Nougat Correction**: This is a series of post processing cleaning step by [Nougat](https://github.com/facebookresearch/nougat/blob/main/nougat/postprocessing.py) to make the document markdown compactible.
- **Rule based Correction**: Custom regex based patterns to remove the most commonly occuring errors.
- **LaTeX Correction**: Fixes mathematical equations and notation

## Configuration

### Basic Configuration

```yaml
- name: cleaning
  config:
    ocr_threshold: 0.99
```

### LLM Enhanced Cleaning (Optional)

For the latex correction, the latex components are extracted and passed to an LLM for improvement, the syntax is verified using [pdflatex](https://pypi.org/project/pdflatex/) and then merged back into the document.

To use this module, You need to set the .env key for `OPENROUTER_API_KEY`.

```yaml
- name: cleaning
  config:
    enable_latex_correction: true
```

## Configuration Parameters

### ocr_threshold
- **Type**: Float
- **Default**: `0.99`
- **Description**: This parameter controls what level of similarity is required for two sentences to be considered duplicate.

### min_words
- **Type**: Int
- **Default**: 2
- **Description**: This parameter defines the minimum number of words a sentence should have for the duplication process. Higher the value, the more accurate the duplicate ocr segments are removed.

### enable_latex_correction
- **Type**: Boolean
- **Default**: `false`
- **Description**: Use LLM to fix latex formulas and tables

### openrouter_model
- **Type**: String
- **Default**: `anthropic/claude-3-haiku`
- **Description**: The model to be used for latex correction

### debug
- **Type**: Boolean
- **Default**: `false`
- **Description**: To enable debug output

## Next Steps

- Learn about [PII removal](pii-removal.md)
- Configure [metadata extraction](metadata-extraction.md)
- Set up [document export](export.md)