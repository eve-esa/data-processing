# EVE Pipeline

[![ESA](https://img.shields.io/badge/ESA-Φ--lab-blue.svg)](https://philab.esa.int/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A high-performance, modular library for extracting, deduplicating, cleaning, anonymizing, and exporting large-scale Earth science and Earth observation datasets. This is part of the Earth Virtual Expert (EVE) initiative—an open-science program funded by the European Space Agency's Φ-lab.

## 🌍 Earth Virtual Expert (EVE)

**Earth Virtual Expert (EVE)** aims to advance the use of Large Language Models (LLMs) within the Earth Observation (EO) and Earth Science (ES) community.

- **Website**: https://eve.philab.esa.int/
- **HuggingFace**: https://huggingface.co/eve-esa
- **GitHub**: https://github.com/eve-esa

## Features

### Extraction
- Supports PDF, HTML, XML, Markdown and nested folder structures
- Automatically detects file formats unless explicitly specified

### Deduplication
- Performs exact matching using SHA-256 checksum
- Supports LSH based near-duplicate detection with configurable:
  - Shingle size
  - Permutations
  - Similarity threshold

### Cleaning
- Removes irregularities and noise artifacts
- Corrects LaTeX equations and tables using LLM assistance

### PII Removal
- Automatically masks Names and Emails using the Presidio framework
- Configurable detection patterns

### Metadata Extraction
- Extracts Title, Authors, DOI, URL, Year, Journal, and Citation Count
- PDF-based extraction using MonkeyOCR integration
- Support for HTML and other formats

### Export
- Saves processed content in multiple formats (default: Markdown)

## Quick Start

1. **Install the packages**

    ```bash
    uv sync
    ```

2. **Configure the pipeline** (`config.yaml`)

    ```yaml
    pipeline:
      batch_size: 10
      inputs:
        path: "input_dir"
      stages:
        - name: extraction
          config: { format: "xml"}
        - name: duplication
          config: { method: "lsh", shingle_size: 3, num_perm: 128, threshold: 0.8 }
        - name: pii
          config: { url: "http://127.0.0.1:8000" }
        - name: export
          config: { format: "md", destination: "output/files"}
    ```

3. **Run the pipeline**

    ```bash
    eve run
    ```

## Documentation

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quick-start.md)
<!-- - [Examples](examples/) -->

## Contributing

We welcome contributions! Please open an issue or submit a pull request on GitHub to help improve the pipeline.

## License

This project is released under the Apache 2.0 License - see the [LICENSE](https://github.com/eve-esa/eve-pipeline/blob/main/LICENSE) file for more details.

## Acknowledgments

This project is supported by the European Space Agency (ESA) Φ-lab through the Large Language Model for Earth Observation and Earth Science project, as part of the Foresight Element within FutureEO Block 4 programme.