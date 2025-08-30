# Eve Data Processing Pipeline
(Design Document - https://docs.google.com/document/d/13sbBslvo7HGYX7pooL8tkjwldvrkVOsFyIZRJVTWLZg/edit?usp=sharing)


## 🏗️ Architecture

The pipeline consists of 5 main stages:

1. **Extraction**: Multi-format document processing
2. **Cleaning**: 5-check cleaning process
3. **PII Removal**: Sensitive data anonymization  
4. **Deduplication**: Exact and approximate duplicate detection
5. **LaTeX Correction**: Mathematical notation fixes using GPT-4o-mini
