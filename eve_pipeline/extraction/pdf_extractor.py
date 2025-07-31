"""PDF extraction using multiple methods (Nougat, Marker, PyPDF)."""

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile
import subprocess

from eve_pipeline.extraction.base import ExtractorBase


class PDFExtractor(ExtractorBase):
    """PDF content extractor supporting multiple extraction methods."""
    
    def __init__(
        self,
        method: str = "nougat",
        nougat_checkpoint: Optional[str] = None,
        batch_size: int = 4,
        **kwargs,
    ) -> None:
        """Initialize PDF extractor.
        
        Args:
            method: Extraction method ('nougat', 'marker', 'pypdf').
            nougat_checkpoint: Path to Nougat model checkpoint.
            batch_size: Batch size for processing.
            **kwargs: Additional configuration.
        """
        super().__init__(
            supported_formats=["pdf"],
            output_format="markdown",
            **kwargs,
        )
        self.method = method.lower()
        self.nougat_checkpoint = nougat_checkpoint or os.getenv("NOUGAT_CHECKPOINT")
        self.batch_size = batch_size
        
        # Initialize the selected method
        self._initialize_extractor()
    
    def _initialize_extractor(self) -> None:
        """Initialize the selected extraction method."""
        if self.method == "nougat":
            self._initialize_nougat()
        elif self.method == "marker":
            self._initialize_marker()
        elif self.method == "pypdf":
            self._initialize_pypdf()
        else:
            raise ValueError(f"Unsupported extraction method: {self.method}")
    
    def _initialize_nougat(self) -> None:
        """Initialize Nougat OCR."""
        try:
            from nougat import NougatModel
            from nougat.utils.checkpoint import get_checkpoint
            from nougat.utils.device import move_to_device
            
            checkpoint = self.nougat_checkpoint or get_checkpoint()
            if not checkpoint:
                raise RuntimeError(
                    "Nougat checkpoint not found. Set NOUGAT_CHECKPOINT environment variable."
                )
            
            self.nougat_model = NougatModel.from_pretrained(checkpoint)
            self.nougat_model = move_to_device(self.nougat_model, cuda=self.batch_size > 0)
            self.nougat_model.eval()
            
            self.logger.info(f"Initialized Nougat with checkpoint: {checkpoint}")
            
        except ImportError as e:
            raise ImportError(f"Nougat not available: {e}")
    
    def _initialize_marker(self) -> None:
        """Initialize Marker PDF processor."""
        try:
            # Marker is typically used via command line or API
            # Check if marker-pdf is available
            result = subprocess.run(
                ["marker-pdf", "--help"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError("marker-pdf command not found")
            
            self.logger.info("Initialized Marker PDF extractor")
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            raise RuntimeError(f"Marker not available: {e}")
    
    def _initialize_pypdf(self) -> None:
        """Initialize PyPDF extractor."""
        try:
            import pypdf
            self.logger.info("Initialized PyPDF extractor")
        except ImportError as e:
            raise ImportError(f"PyPDF not available: {e}")
    
    def extract_content(self, file_path: Path) -> str:
        """Extract content from PDF file.
        
        Args:
            file_path: Path to PDF file.
            
        Returns:
            Extracted content as markdown.
        """
        if self.method == "nougat":
            return self._extract_with_nougat(file_path)
        elif self.method == "marker":
            return self._extract_with_marker(file_path)
        elif self.method == "pypdf":
            return self._extract_with_pypdf(file_path)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
    
    def _extract_with_nougat(self, file_path: Path) -> str:
        """Extract using Nougat OCR."""
        try:
            import pypdfium2
            import torch
            from functools import partial
            from nougat.dataset.rasterize import rasterize_paper
            from nougat.utils.dataset import ImageDataset
            from nougat.postprocessing import markdown_compatible
            from tqdm import tqdm
            
            # Read PDF
            with open(file_path, "rb") as f:
                pdfbin = f.read()
            
            pdf = pypdfium2.PdfDocument(pdfbin)
            pages = list(range(len(pdf)))
            
            # Rasterize pages
            images = rasterize_paper(pdf, pages=pages)
            
            if not images:
                return ""
            
            # Create dataset
            dataset = ImageDataset(
                images,
                partial(self.nougat_model.encoder.prepare_input, random_padding=False),
            )
            
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                pin_memory=True,
                shuffle=False,
            )
            
            predictions = [""] * len(pages)
            
            # Process batches
            for idx, sample in enumerate(dataloader):
                if sample is None:
                    continue
                
                model_output = self.nougat_model.inference(image_tensors=sample)
                
                for j, output in enumerate(model_output["predictions"]):
                    page_idx = idx * self.batch_size + j
                    if page_idx < len(predictions):
                        predictions[page_idx] = markdown_compatible(output)
            
            return "".join(predictions).strip()
            
        except Exception as e:
            self.logger.error(f"Nougat extraction failed: {e}")
            raise
    
    def _extract_with_marker(self, file_path: Path) -> str:
        """Extract using Marker."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = Path(temp_dir) / "output.md"
                
                # Run marker-pdf command
                result = subprocess.run(
                    [
                        "marker-pdf",
                        str(file_path),
                        str(output_path),
                        "--batch_multiplier", str(self.batch_size),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                )
                
                if result.returncode != 0:
                    raise RuntimeError(f"Marker failed: {result.stderr}")
                
                if output_path.exists():
                    return output_path.read_text(encoding="utf-8")
                else:
                    raise RuntimeError("Marker output file not found")
                    
        except Exception as e:
            self.logger.error(f"Marker extraction failed: {e}")
            raise
    
    def _extract_with_pypdf(self, file_path: Path) -> str:
        """Extract using PyPDF."""
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(str(file_path))
            text_parts = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(f"## Page {page_num + 1}\n\n{text}\n\n")
                except Exception as e:
                    self.logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                    continue
            
            if not text_parts:
                return ""
            
            # Add document header
            header = self._create_markdown_header(file_path)
            return header + "".join(text_parts)
            
        except Exception as e:
            self.logger.error(f"PyPDF extraction failed: {e}")
            raise
    
    def get_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get PDF metadata.
        
        Args:
            file_path: Path to PDF file.
            
        Returns:
            Dictionary with metadata.
        """
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(str(file_path))
            metadata = {
                "num_pages": len(reader.pages),
                "file_size": file_path.stat().st_size,
            }
            
            if reader.metadata:
                metadata.update({
                    "title": reader.metadata.get("/Title", ""),
                    "author": reader.metadata.get("/Author", ""),
                    "subject": reader.metadata.get("/Subject", ""),
                    "creator": reader.metadata.get("/Creator", ""),
                    "producer": reader.metadata.get("/Producer", ""),
                })
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata: {e}")
            return {"num_pages": 0, "file_size": file_path.stat().st_size}