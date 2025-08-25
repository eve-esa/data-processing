"""PII removal client for batch processing."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Union

import requests
from tqdm import tqdm

from eve_pipeline.core.base import ProcessorResult, ProcessorStatus


class PIIClient:
    """Client for batch PII removal processing."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        max_workers: int = 1,
        timeout: int = 120,
    ) -> None:
        """Initialize PII client.

        Args:
            server_url: URL of the PII removal server.
            max_workers: Maximum number of concurrent workers.
            timeout: Request timeout in seconds.
        """
        self.server_url = server_url.rstrip("/")
        self.max_workers = max_workers
        self.timeout = timeout
        self.predict_url = f"{self.server_url}/predict"
        self.logger = logging.getLogger("PIIClient")

    def remove_pii(
        self,
        text: str,
        entities: Optional[list[str]] = None,
        score_threshold: float = 0.35,
        return_analysis: bool = False,
    ) -> dict[str, Any]:
        """Remove PII from text using the server.

        Args:
            text: Input text.
            entities: List of entity types to detect.
            score_threshold: Minimum confidence score.
            return_analysis: Whether to return detailed analysis.

        Returns:
            Dictionary with processing results.
        """
        if entities is None:
            entities = ["PERSON", "EMAIL_ADDRESS"]

        payload = {
            "text": text,
            "entities": entities,
            "score_threshold": score_threshold,
            "return_analysis": return_analysis,
        }

        try:
            response = requests.post(
                self.predict_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            return {
                "anonymized_text": text,
                "entities_found": [],
                "processing_time": 0.0,
                "success": False,
                "error_message": f"Request failed: {e!s}",
            }

    def process_file(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        entities: Optional[list[str]] = None,
        score_threshold: float = 0.35,
        return_analysis: bool = False,
        skip_existing: bool = False,
    ) -> ProcessorResult:
        """Process a single file for PII removal.

        Args:
            file_path: Path to input file.
            output_path: Optional path for output file.
            entities: List of entity types to detect.
            score_threshold: Minimum confidence score.
            return_analysis: Whether to return detailed analysis.
            skip_existing: Whether to skip if output exists.

        Returns:
            ProcessorResult with processing outcome.
        """
        start_time = time.time()
        file_path = Path(file_path)

        # Check if should skip
        if skip_existing and output_path and Path(output_path).exists():
            return ProcessorResult(
                status=ProcessorStatus.SKIPPED,
                input_path=file_path,
                output_path=output_path,
                processing_time=time.time() - start_time,
                metadata={"skip_reason": "Output file already exists"},
            )

        try:
            # Read input file
            content = self._read_file(file_path)

            # Process content
            result = self.remove_pii(
                content, entities, score_threshold, return_analysis,
            )

            processing_time = time.time() - start_time

            if not result.get("success", False):
                return ProcessorResult(
                    status=ProcessorStatus.FAILED,
                    input_path=file_path,
                    processing_time=processing_time,
                    error_message=result.get("error_message", "PII removal failed"),
                )

            # Save output if path provided
            if output_path and result.get("anonymized_text"):
                output_path = Path(output_path)
                self._write_file(output_path, result["anonymized_text"])

            return ProcessorResult(
                status=ProcessorStatus.SUCCESS,
                input_path=file_path,
                output_path=output_path,
                content=result["anonymized_text"],
                processing_time=processing_time,
                metadata={
                    "entities_found": result.get("entities_found", []),
                    "server_processing_time": result.get("processing_time", 0),
                    "entities_removed": len(result.get("entities_found", [])),
                },
            )

        except Exception as e:
            return ProcessorResult(
                status=ProcessorStatus.FAILED,
                input_path=file_path,
                processing_time=time.time() - start_time,
                error_message=f"File processing failed: {e!s}",
            )

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        entities: Optional[list[str]] = None,
        score_threshold: float = 0.35,
        return_analysis: bool = True,
        file_extensions: Optional[list[str]] = None,
        skip_existing: bool = True,
    ) -> dict[str, Any]:
        """Process all files in a directory for PII removal.

        Args:
            input_dir: Path to input directory.
            output_dir: Path to output directory.
            entities: List of entity types to detect.
            score_threshold: Minimum confidence score.
            return_analysis: Whether to return detailed analysis.
            file_extensions: List of file extensions to process.
            skip_existing: Whether to skip existing output files.

        Returns:
            Dictionary containing processing results and statistics.
        """
        if file_extensions is None:
            file_extensions = [".md"]

        input_path = Path(input_dir)

        if not input_path.exists():
            return {
                "success": False,
                "error_message": f"Input directory does not exist: {input_dir}",
                "files_processed": 0,
                "results": [],
            }

        if output_dir is None:
            output_dir = f"{input_dir}_pii_removed"

        output_path = Path(output_dir)

        # Find all files to process
        file_paths = []
        output_paths = []

        for file_path in input_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                file_paths.append(file_path)

                # Create output path maintaining directory structure
                relative_path = file_path.relative_to(input_path)
                output_file_path = output_path / relative_path
                output_paths.append(output_file_path)

        if not file_paths:
            return {
                "success": True,
                "message": f"No files found with extensions {file_extensions} in {input_dir}",
                "files_processed": 0,
                "results": [],
            }

        self.logger.info(f"Found {len(file_paths)} files to process")
        self.logger.info(f"Processing files from: {input_dir}")
        self.logger.info(f"Saving results to: {output_dir}")

        # Process files
        results = self._process_batch(
            file_paths=file_paths,
            output_paths=output_paths,
            entities=entities,
            score_threshold=score_threshold,
            return_analysis=return_analysis,
            skip_existing=skip_existing,
        )

        # Generate statistics
        successful_files = [r for r in results if r.is_success]
        failed_files = [r for r in results if r.is_failed]
        skipped_files = [r for r in results if r.is_skipped]

        total_entities_found = sum(
            len(r.metadata.get("entities_found", []))
            for r in successful_files
            if r.metadata
        )

        total_processing_time = sum(r.processing_time for r in results)

        return {
            "success": True,
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "files_processed": len(successful_files),
            "files_skipped": len(skipped_files),
            "files_failed": len(failed_files),
            "total_files": len(file_paths),
            "total_entities_found": total_entities_found,
            "total_processing_time": total_processing_time,
            "file_extensions_processed": file_extensions,
            "skip_existing_enabled": skip_existing,
            "results": results,
        }

    def _process_batch(
        self,
        file_paths: list[Path],
        output_paths: list[Path],
        entities: Optional[list[str]],
        score_threshold: float,
        return_analysis: bool,
        skip_existing: bool,
    ) -> list[ProcessorResult]:
        """Process a batch of files."""
        results = []

        if self.max_workers > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        self.process_file,
                        file_path,
                        output_path,
                        entities,
                        score_threshold,
                        return_analysis,
                        skip_existing,
                    ): file_path
                    for file_path, output_path in zip(file_paths, output_paths)
                }

                for future in tqdm(
                    as_completed(future_to_file),
                    total=len(file_paths),
                    desc="Processing files",
                ):
                    results.append(future.result())
        else:
            # Sequential processing
            for file_path, output_path in tqdm(
                zip(file_paths, output_paths),
                total=len(file_paths),
                desc="Processing files",
            ):
                result = self.process_file(
                    file_path, output_path, entities, score_threshold,
                    return_analysis, skip_existing,
                )
                results.append(result)

        return results

    def _read_file(self, file_path: Path) -> str:
        """Read file with encoding detection."""
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise Exception(f"Cannot decode file {file_path} with any supported encoding")

    def _write_file(self, file_path: Path, content: str) -> None:
        """Write content to file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def health_check(self) -> bool:
        """Check if the PII removal server is healthy.

        Returns:
            True if server is responding, False otherwise.
        """
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
