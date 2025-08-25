"""CSV file extraction."""

import csv
from pathlib import Path
from typing import Any, Optional

from eve_pipeline.extraction.base import ExtractorBase


class CSVExtractor(ExtractorBase):
    """CSV file extractor."""

    def __init__(
        self,
        delimiter: str = ",",
        max_rows: Optional[int] = None,
        include_headers: bool = True,
        **kwargs,
    ) -> None:
        """Initialize CSV extractor.

        Args:
            delimiter: CSV delimiter character.
            max_rows: Maximum number of rows to extract (None for all).
            include_headers: Whether to include column headers.
            **kwargs: Additional configuration.
        """
        super().__init__(
            supported_formats=["csv", "tsv"],
            output_format="markdown",
            **kwargs,
        )
        self.delimiter = delimiter
        self.max_rows = max_rows
        self.include_headers = include_headers

    def extract_content(self, file_path: Path) -> str:
        """Extract content from CSV file.

        Args:
            file_path: Path to CSV file.

        Returns:
            CSV content as markdown table.
        """
        try:
            # Auto-detect delimiter if TSV
            delimiter = self.delimiter
            if file_path.suffix.lower() == ".tsv":
                delimiter = "\t"

            # Read CSV content
            rows = self._read_csv_file(file_path, delimiter)

            if not rows:
                raise RuntimeError("Empty CSV file")

            # Convert to markdown table
            markdown_content = self._convert_to_markdown_table(rows, file_path)

            return markdown_content

        except Exception as e:
            self.logger.error(f"CSV extraction failed: {e}")
            raise

    def _read_csv_file(self, file_path: Path, delimiter: str) -> list[list[str]]:
        """Read CSV file with encoding detection.

        Args:
            file_path: Path to CSV file.
            delimiter: CSV delimiter.

        Returns:
            List of rows, where each row is a list of strings.
        """
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, encoding=encoding, newline="") as f:
                    # Try to detect dialect
                    sample = f.read(1024)
                    f.seek(0)

                    try:
                        dialect = csv.Sniffer().sniff(sample, delimiters=delimiter)
                        reader = csv.reader(f, dialect)
                    except csv.Error:
                        # Fallback to simple reader
                        reader = csv.reader(f, delimiter=delimiter)

                    rows = []
                    for i, row in enumerate(reader):
                        if self.max_rows and i >= self.max_rows:
                            break
                        rows.append(row)

                    return rows

            except UnicodeDecodeError:
                continue

        raise RuntimeError(f"Cannot decode CSV file {file_path}")

    def _convert_to_markdown_table(self, rows: list[list[str]], file_path: Path) -> str:
        """Convert CSV rows to markdown table.

        Args:
            rows: CSV rows.
            file_path: Path to original file.

        Returns:
            Markdown table.
        """
        if not rows:
            return ""

        # Add header
        markdown_content = self._create_markdown_header(file_path)

        # Handle headers
        if self.include_headers and rows:
            headers = rows[0]
            data_rows = rows[1:]
        else:
            # Generate column headers
            max_cols = max(len(row) for row in rows) if rows else 0
            headers = [f"Column {i+1}" for i in range(max_cols)]
            data_rows = rows

        if not headers:
            return markdown_content + "Empty CSV file.\n"

        # Create markdown table
        markdown_content += "| " + " | ".join(headers) + " |\n"
        markdown_content += "|" + "|".join([" --- "] * len(headers)) + "|\n"

        for row in data_rows:
            # Pad row to match header length
            padded_row = row + [""] * (len(headers) - len(row))
            # Escape pipes in cell content
            escaped_row = [cell.replace("|", "\\|") for cell in padded_row[:len(headers)]]
            markdown_content += "| " + " | ".join(escaped_row) + " |\n"

        # Add summary
        summary = f"\n\n**Summary:** {len(data_rows)} rows, {len(headers)} columns"
        if self.max_rows and len(data_rows) >= self.max_rows:
            summary += f" (limited to first {self.max_rows} rows)"
        markdown_content += summary

        return markdown_content

    def get_csv_metadata(self, file_path: Path) -> dict[str, Any]:
        """Get CSV metadata.

        Args:
            file_path: Path to CSV file.

        Returns:
            Dictionary with metadata.
        """
        try:
            # Detect delimiter
            delimiter = self.delimiter
            if file_path.suffix.lower() == ".tsv":
                delimiter = "\t"

            rows = self._read_csv_file(file_path, delimiter)

            if not rows:
                return {"file_size": file_path.stat().st_size, "rows": 0, "columns": 0}

            # Calculate statistics
            max_cols = max(len(row) for row in rows)
            min_cols = min(len(row) for row in rows)

            metadata = {
                "file_size": file_path.stat().st_size,
                "rows": len(rows),
                "columns": max_cols,
                "delimiter": delimiter,
                "encoding": self._detect_encoding(file_path),
                "consistent_columns": min_cols == max_cols,
            }

            # Sample of first few cells for preview
            if rows:
                metadata["preview"] = rows[0][:5]  # First 5 columns of first row

            return metadata

        except Exception as e:
            self.logger.warning(f"Failed to extract CSV metadata: {e}")
            return {"file_size": file_path.stat().st_size}

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding.

        Args:
            file_path: Path to file.

        Returns:
            Detected encoding.
        """
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, encoding=encoding) as f:
                    f.read(1024)
                return encoding
            except UnicodeDecodeError:
                continue

        return "unknown"
