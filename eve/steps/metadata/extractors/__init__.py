"""
Metadata Extractors Package.

This package contains specialized extractors for different document formats.
Each extractor implements the BaseMetadataExtractor interface and provides
format-specific logic for extracting metadata.

Extractor Classes:
- BaseMetadataExtractor: Abstract base class with shared utilities
- PdfMetadataExtractor: PDF documents (pdf2bib + direct reading)
- HtmlMetadataExtractor: HTML/web pages (title, meta tags, structured data)  
- ScholarMetadataExtractor: Academic papers via Google Scholar search

Design Principles:
- Inheritance from BaseMetadataExtractor for consistency
- Shared utilities to eliminate code duplication
- Graceful degradation with multiple fallback strategies
- Async/await support for non-blocking operations
- Comprehensive error handling and logging

Shared Utilities (BaseMetadataExtractor):
- Document format validation
- Field mapping with optional cleaning
- Title extraction with filename fallback
- Author name processing and normalization
- Extraction method tracking
- Metadata finalization and debug logging

Usage:
    from eve.steps.metadata.extractors.pdf_extractor import PdfMetadataExtractor
    
    extractor = PdfMetadataExtractor(debug=True)
    metadata = await extractor.extract_metadata(document)
"""
