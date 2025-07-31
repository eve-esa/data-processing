"""XML content extraction."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

from eve_pipeline.extraction.base import ExtractorBase


class XMLExtractor(ExtractorBase):
    """XML content extractor."""
    
    def __init__(
        self,
        preserve_structure: bool = True,
        extract_attributes: bool = False,
        **kwargs,
    ) -> None:
        """Initialize XML extractor.
        
        Args:
            preserve_structure: Whether to preserve XML structure in output.
            extract_attributes: Whether to extract element attributes.
            **kwargs: Additional configuration.
        """
        super().__init__(
            supported_formats=["xml"],
            output_format="markdown",
            **kwargs,
        )
        self.preserve_structure = preserve_structure
        self.extract_attributes = extract_attributes
    
    def extract_content(self, file_path: Path) -> str:
        """Extract content from XML file.
        
        Args:
            file_path: Path to XML file.
            
        Returns:
            Extracted content as markdown.
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            if self.preserve_structure:
                content = self._extract_structured_content(root)
            else:
                content = self._extract_text_content(root)
            
            # Clean up content
            content = self._clean_content(content)
            
            # Add header
            header = self._create_markdown_header(file_path)
            return header + content
            
        except ET.ParseError as e:
            self.logger.error(f"XML parsing error: {e}")
            raise RuntimeError(f"Invalid XML file: {e}")
        except Exception as e:
            self.logger.error(f"XML extraction failed: {e}")
            raise
    
    def _extract_text_content(self, element: ET.Element) -> str:
        """Extract all text content from XML tree recursively.
        
        Args:
            element: XML element.
            
        Returns:
            Extracted text content.
        """
        texts = []
        
        # Add element text
        if element.text and element.text.strip():
            texts.append(element.text.strip())
        
        # Process children recursively
        for child in element:
            child_text = self._extract_text_content(child)
            if child_text:
                texts.append(child_text)
        
        # Add tail text
        if element.tail and element.tail.strip():
            texts.append(element.tail.strip())
        
        return " ".join(texts)
    
    def _extract_structured_content(self, element: ET.Element, level: int = 0) -> str:
        """Extract content while preserving structure.
        
        Args:
            element: XML element.
            level: Current nesting level.
            
        Returns:
            Structured content as markdown.
        """
        content_parts = []
        
        # Add element as header if it has a meaningful tag
        tag_name = self._clean_tag_name(element.tag)
        if tag_name and level < 6:  # Limit header levels
            header_level = min(level + 1, 6)
            header = "#" * header_level + f" {tag_name}"
            
            # Add attributes if requested
            if self.extract_attributes and element.attrib:
                attrs = ", ".join(f"{k}={v}" for k, v in element.attrib.items())
                header += f" ({attrs})"
            
            content_parts.append(header)
        
        # Add element text
        if element.text and element.text.strip():
            content_parts.append(element.text.strip())
        
        # Process children
        for child in element:
            child_content = self._extract_structured_content(child, level + 1)
            if child_content:
                content_parts.append(child_content)
        
        # Add tail text
        if element.tail and element.tail.strip():
            content_parts.append(element.tail.strip())
        
        return "\n\n".join(content_parts)
    
    def _clean_tag_name(self, tag: str) -> str:
        """Clean XML tag name for use as header.
        
        Args:
            tag: XML tag name.
            
        Returns:
            Cleaned tag name.
        """
        # Remove namespace prefixes
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        
        # Convert to title case
        tag = tag.replace("_", " ").replace("-", " ").title()
        
        # Skip common structural tags
        skip_tags = {"Root", "Document", "Body", "Content", "Text"}
        if tag in skip_tags:
            return ""
        
        return tag
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content.
        
        Args:
            content: Raw extracted content.
            
        Returns:
            Cleaned content.
        """
        # Remove excessive whitespace
        content = re.sub(r"\s+", " ", content)
        
        # Remove excessive newlines
        content = re.sub(r"\n{3,}", "\n\n", content)
        
        # Clean up around headers
        content = re.sub(r"\n\s*#", "\n#", content)
        
        return content.strip()
    
    def get_xml_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Get XML metadata.
        
        Args:
            file_path: Path to XML file.
            
        Returns:
            Dictionary with metadata.
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            metadata = {
                "root_tag": root.tag,
                "file_size": file_path.stat().st_size,
                "element_count": len(list(root.iter())),
                "max_depth": self._get_max_depth(root),
            }
            
            # Extract namespace information
            namespaces = {}
            for elem in root.iter():
                if "}" in elem.tag:
                    ns, tag = elem.tag.split("}", 1)
                    ns = ns.lstrip("{")
                    if ns not in namespaces:
                        namespaces[ns] = tag
            
            if namespaces:
                metadata["namespaces"] = namespaces
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to extract XML metadata: {e}")
            return {"file_size": file_path.stat().st_size}
    
    def _get_max_depth(self, element: ET.Element, current_depth: int = 0) -> int:
        """Get maximum depth of XML tree.
        
        Args:
            element: XML element.
            current_depth: Current depth.
            
        Returns:
            Maximum depth.
        """
        if not list(element):
            return current_depth
        
        return max(
            self._get_max_depth(child, current_depth + 1)
            for child in element
        )