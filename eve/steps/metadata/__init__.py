"""
EVE Pipeline Metadata Extraction Module.

This package provides comprehensive metadata extraction capabilities for the EVE pipeline.
It supports multiple document formats and uses specialized extractors to extract
bibliographic information, titles, authors, and other document metadata.

Module Structure:
- metadata_step.py: Main orchestrator for metadata extraction
- extractors/: Format-specific metadata extractors
  - base_extractor.py: Shared utilities and base class
  - pdf_extractor.py: PDF metadata extraction (pdf2bib + direct reading)
  - html_extractor.py: HTML metadata extraction (title, meta tags, structured data)
  - scholar_extractor.py: Google Scholar search for academic papers

Key Features:
- Multi-format support (PDF, HTML, TXT, MD)
- Graceful fallback mechanisms
- Google Scholar integration for text documents
- Parallel processing for performance
- Comprehensive metadata export
- Configurable extraction methods

Usage:
    from eve.steps.metadata.metadata_step import MetadataStep
    
    config = {
        "enabled_formats": ["pdf", "html"],
        "enable_scholar_search": True,
        "export_metadata": True
    }
    
    metadata_step = MetadataStep(config)
    enriched_documents = await metadata_step.execute(documents)
"""
