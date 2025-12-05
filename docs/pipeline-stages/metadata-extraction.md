# Metadata Extraction Stage

The metadata extraction stage automatically identifies and extracts structured metadata from documents.


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

Setup MonkeyOCR using the bash file under the `\server` directory.

1. We first extract text from the first page of the PDF files using MonkeyOCR. The doi and the title are usually present within the first page of the document.
2. We extract dois using handwritten regex patterns
    - if the file is from arXiv, we invoke the [arXiv API](https://info.arxiv.org/help/api/basics.html) to extract metadata.
    - if the file is from other publishers, we invoke the [crossref API](https://www.crossref.org/) to extract metadata.
3. Fallback - if doi is not present, we extract the title and then invoke the crossref API using the title to extract the metadata.

### Other format Extraction

For other documents like HTML, TXT, JSON, the extractor uses handwritten regex patterns to extract the document title and the URL of the page.

```yaml
- name: metadata
  config:
    enabled_formats: ["pdf", "html", "txt", "md"]
```

## Configuration Parameters

### enabled_formats
- **Type**: List
- **Default**: `["pdf", "html", "txt", "md"]`
- **Description**: The list of file formats to process.

### export_metadata
- **Type**: Boolean
- **Default**: `true`
- **Description**: Whether to export metadata to JSON file.

### metadata_destination
- **Type**: String
- **Default**: `./output`
- **Description**: Directory to save metadata file

## Next Steps

- Learn about [document export](export.md)