# Pipeline Stages

This section documents all the available pipeline stages for processing documents.

## Extraction Stage

Extracts content from various document formats.

::: eve.steps.extraction.extract_step
    :members:
    :show-inheritance:

### Extractors

#### PDF Extractor
::: eve.steps.extraction.pdfs
    :members:

#### HTML Extractor
::: eve.steps.extraction.htmls
    :members:

#### XML Extractor
::: eve.steps.extraction.xmls
    :members:

#### Markdown Extractor
::: eve.steps.extraction.markdown
    :members:

## Deduplication Stage

Removes duplicate and near-duplicate documents.

::: eve.steps.dedup.dedup_step
    :members:
    :show-inheritance:

### Deduplication Methods

#### Exact Duplicates
::: eve.steps.dedup.exact_duplicates
    :members:

#### MinHash LSH
::: eve.steps.dedup.minhash
    :members:

## Cleaning Stage

Cleans and improves document quality.

::: eve.steps.cleaning.cleaning_step
    :members:
    :show-inheritance:

### Cleaning Components

#### Processors
::: eve.steps.cleaning.processors
    :members:

#### Nougat Helpers
::: eve.steps.cleaning.nougat_helpers
    :members:

## PII Removal Stage

Removes personally identifiable information from documents.

::: eve.steps.pii.pii_step
    :members:
    :show-inheritance:

## Metadata Extraction Stage

Extracts structured metadata from documents.

::: eve.steps.metadata.metadata_step
    :members:
    :show-inheritance:

### Metadata Extractors

#### HTML Metadata Extractor
::: eve.steps.metadata.extractors.html_extractor
    :members:

#### PDF Metadata Extractor
::: eve.steps.metadata.extractors.pdf_extractor
    :members:

## Export Stage

Saves processed documents to output formats.

::: eve.steps.export.export_step
    :members:
    :show-inheritance: