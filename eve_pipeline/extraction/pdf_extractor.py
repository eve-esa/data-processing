"""PDF extraction using multiple methods (Nougat, Marker, PyPDF)."""

import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

from eve_pipeline.extraction.base import ExtractorBase


class PDFExtractor(ExtractorBase):
    """PDF content extractor supporting multiple extraction methods.
    
    Supported methods:
    - nougat: OCR-based extraction using Nougat model (good for academic papers with equations)
    - marker: Layout-aware extraction using Marker (good for complex layouts and tables)
    
    The extractor automatically handles both local files and S3 paths.
    
    Args:
        method: Extraction method ('nougat' or 'marker'). Default is 'nougat'.
        nougat_checkpoint: Path to Nougat model checkpoint (for nougat method).
        batch_size: Batch size for processing multiple pages.
        
    Example:
        # Using Nougat (default)
        extractor = PDFExtractor()
        result = extractor.extract_content("paper.pdf")
        
        # Using Marker for better layout preservation
        extractor = PDFExtractor(method="marker", batch_size=2)
        result = extractor.extract_content("document.pdf")
    """

    def __init__(
        self,
        method: str = "nougat",
        nougat_checkpoint: Optional[str] = None,
        batch_size: int = 4,
        **kwargs,
    ) -> None:
        """Initialize PDF extractor.

        Args:
            method: Extraction method ('nougat', 'marker').
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
        else:
            raise ValueError(f"Unsupported extraction method: {self.method}. Supported methods: 'nougat', 'marker'")

    def _initialize_nougat(self) -> None:
        """Initialize Nougat OCR."""
        try:
            from nougat import NougatModel
            from nougat.utils.checkpoint import get_checkpoint
            from nougat.utils.device import move_to_device

            checkpoint = self.nougat_checkpoint or get_checkpoint()
            if not checkpoint:
                raise RuntimeError(
                    "Nougat checkpoint not found. Set NOUGAT_CHECKPOINT environment variable.",
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
            # Check if marker package is available with the new API
            import importlib.util
            if importlib.util.find_spec("marker") is None:
                raise ImportError("Marker package not available")
            
            # Verify the required modules exist
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
            
            # Converter will be initialized on first use for better startup performance
            self.logger.info("Initialized Marker PDF extractor")

        except ImportError as e:
            raise ImportError(f"Marker package not available. Install with: pip install marker-pdf")



    def extract_content(self, file_path: Union[str, Path]) -> str:
        """Extract content from PDF file.

        Args:
            file_path: Path to PDF file (local or S3).

        Returns:
            Extracted content as markdown.
        """
        if self.method == "nougat":
            return self._extract_with_nougat(file_path)
        elif self.method == "marker":
            return self._extract_with_marker(file_path)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}. Supported methods: 'nougat', 'marker'")

    def _extract_with_nougat(self, file_path: Union[str, Path]) -> str:
        """Extract using Nougat OCR."""
        try:
            from functools import partial

            import pypdfium2
            import torch
            from nougat.dataset.rasterize import rasterize_paper
            from nougat.postprocessing import markdown_compatible
            from nougat.utils.dataset import ImageDataset

            # Read PDF using storage backend for S3 support
            if isinstance(file_path, str) and file_path.startswith('s3://'):
                # Use storage backend for S3 files
                from eve_pipeline.storage.factory import StorageFactory
                storage = StorageFactory.get_storage_for_path(file_path, **self.storage_config)
                pdfbin = storage.read_bytes(file_path)
            else:
                # Read local file directly
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

    def _extract_with_marker(self, file_path: Union[str, Path]) -> str:
        """Extract using Marker."""
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
            
            # Initialize converter (models are loaded lazily)
            if not hasattr(self, '_marker_converter'):
                self.logger.info("Initializing Marker converter...")
                self._marker_converter = PdfConverter(
                    artifact_dict=create_model_dict(),
                )
                self.logger.info("Marker converter initialized")

            if isinstance(file_path, str) and file_path.startswith('s3://'):
                # Handle S3 files by downloading to temporary location
                with tempfile.TemporaryDirectory() as temp_dir:
                    from eve_pipeline.storage.factory import StorageFactory
                    storage = StorageFactory.get_storage_for_path(file_path, **self.storage_config)
                    pdf_data = storage.read_bytes(file_path)

                    temp_pdf_path = Path(temp_dir) / "temp.pdf"
                    with open(temp_pdf_path, "wb") as f:
                        f.write(pdf_data)
                    
                    # Convert PDF to markdown using marker
                    rendered = self._marker_converter(str(temp_pdf_path))
                    text, _, images = text_from_rendered(rendered)
                    
                    return text
            else:
                # Convert PDF to markdown using marker directly
                rendered = self._marker_converter(str(file_path))
                text, _, images = text_from_rendered(rendered)
                
                return text

        except ImportError as e:
            self.logger.error(f"Marker package not available: {e}")
            raise ImportError(f"Marker package not available. Install with: pip install marker-pdf")
        except Exception as e:
            self.logger.error(f"Marker extraction failed: {e}")
            raise


