# Metadata Extraction Stage

The metadata extraction stage automatically identifies and extracts structured metadata from documents, including bibliographic information, document properties, and content characteristics.


## Extracted Metadata Fields

### Document Identification
- **title**: Document title
- **authors**: List of author names
- **doi**: Digital Object Identifier
- **url**: Source URL or link
- **year**: Publication year
- **journal**: Journal or publication name
- **publisher**: Publisher name



## Extraction Methods

### PDF Metadata Extraction

For PDF documents, the extractor uses MonkeyOCR server with crossref to extract the metadata information.

### HTML Metadata Extraction

For HTML documents, the extractor uses regex patterns to extract the metadata information.


## Next Steps

- Learn about [document export](export.md)