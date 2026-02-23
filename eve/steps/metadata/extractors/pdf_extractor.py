"""
PDF metadata extractor is a two stage process.

1. Extract content using monkeyocr.
2. Use crossref to extract metadata from the content.
"""
import os
import re
import json
import asyncio
import httpx
from typing import Dict, Any, Optional
from tqdm.auto import tqdm

from eve.model.document import Document
from eve.logging import get_logger
from eve.common.regex_patterns import doi_regexp, arxiv_regexp, isbn_regexp

MAX_CONCURRENT = 40

main_dir = "server/MonkeyOCR/output" # this should depend on where the monkeyocr outputs are

class PdfMetadataExtractor():

    def __init__(self, debug: bool = False):
        """
        Initialize the PDF metadata extractor.
        
        Args:
            debug: Enable debug logging for detailed extraction information
        """
        self.debug = debug
        self.logger = get_logger(self.__class__.__name__)

    @staticmethod
    def _safe_str(value):
        if isinstance(value, list):
            return value[0] if value else None
        if isinstance(value, dict):
            return json.dumps(value)
        return str(value) if value not in (None, "", []) else None

    @staticmethod
    def _extract_identifier(main_dir, sub_dir):
        md_path = f"{main_dir}/{sub_dir}/{sub_dir}.md"
        try:
            with open(md_path, 'r', encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return "NA", None

        for pattern in doi_regexp:
            match = re.findall(pattern, content, re.I)
            if match:
                return "doi", match[0]

        for pattern in arxiv_regexp:
            match = re.findall(pattern, content, re.I)
            if match:
                return "arxiv", match[0]

        for pattern in isbn_regexp:
            match = re.findall(pattern, content, re.I)
            if match:
                return "isbn", match[0]

        return "NA", None
    
    @staticmethod
    def _extract_title(main_dir, sub_dir):
        json_path = os.path.join(main_dir, sub_dir, f"{sub_dir}_content_list.json")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            candidates = [row["text"] for row in data if row.get("text_level", 0) == 1]
            return max(candidates, key=len) if candidates else "NA"
        except Exception:
            return "NA"

    @staticmethod
    async def _fetch_json(client, url):
        try:
            resp = await client.get(url, timeout=50) # 
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None
    
    async def _fetch_crossref_by_doi(self, client, doi):
        data = await self._fetch_json(client, f"https://api.crossref.org/works/{doi}")
        if not data:
            return None
        item = data.get("message", {})

        return {
            "title": self._safe_str(item.get("title")),
            "authors": ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in item.get("author", [])
                if isinstance(a, dict)
            ) or None,
            "year": self._safe_str(item.get("issued", {}).get("date-parts", [[None]])[0]),
            "publisher": self._safe_str(item.get("publisher")),
            "journal": self._safe_str(item.get("container-title")),
            "pub_url": self._safe_str(item.get("URL")),
            "doi": self._safe_str(item.get("DOI")),
            "citation_count": item.get("is-referenced-by-count")
        }


    async def _fetch_crossref_by_title(self, client, title):
        q = re.sub(r"[\s]+", "+", title)
        data = await self._fetch_json(client, f"https://api.crossref.org/works?query.bibliographic={q}&rows=1")
        if not data:
            return None
        items = data.get("message", {}).get("items", [])
        if not items:
            return None
        item = items[0]

        return {
            "title": self._safe_str(item.get("title")),
            "authors": ", ".join(
                f"{a.get('given', '')} {a.get('family', '')}".strip()
                for a in item.get("author", [])
                if isinstance(a, dict)
            ) or None,
            "year": self._safe_str(item.get("issued", {}).get("date-parts", [[None]])[0]),
            "publisher": self._safe_str(item.get("publisher")),
            "journal": self._safe_str(item.get("container-title")),
            "pub_url": self._safe_str(item.get("URL")),
            "doi": self._safe_str(item.get("DOI")),
        }


    async def fetch_doi_from_arxiv(self, client, arxiv_id):
        data = await self._fetch_json(client, f"https://api.crossref.org/works?query={arxiv_id}")
        if not data:
            return None
        items = data.get("message", {}).get("items", [])
        return self._safe_str(items[0].get("DOI")) if items else None

    async def _extract_metadata(self, sub_dir, main_dir, client, sem):
        async with sem:
            id_type, identifier = self._extract_identifier(main_dir, sub_dir)
            meta = None

            if id_type == "doi":
                meta = await self._fetch_crossref_by_doi(client, identifier)
            elif id_type == "arxiv":
                doi = await self._fetch_doi_from_arxiv(client, identifier)
                if doi:
                    meta = await self._fetch_crossref_by_doi(client, doi)

            title = self._extract_title(main_dir, sub_dir)
            if not meta and title != "NA":
                meta = await self._fetch_crossref_by_title(client, title)

            return {
                "sub_dir": sub_dir,
                "id_type": id_type,
                "identifier": identifier,
                "title_extracted": title,
                "meta": meta,
            }
    
    async def extract_metadata(self, document: Document) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from a PDF document using crossref.
        
        Args:
            document: PDF document to extract metadata from
            
        Returns:
            Dictionary containing extracted metadata with fields:
            - title: Document title
            - authors: List of author names 
            - year: Publication year
            - journal: Publication venue
            - doi: Digital Object Identifier
            - identifier: Document identifier (DOI, arXiv, etc.)
            
            Returns None if document format is invalid
        """

        metadata = {}
        done = set()

        subdirs = [d for d in os.listdir(main_dir)
                if os.path.isdir(os.path.join(main_dir, d)) and d not in done]

        if not subdirs:
            print("All subdirectories already processed.")
            return

        sem = asyncio.Semaphore(MAX_CONCURRENT)
        async with httpx.AsyncClient() as client:
            tasks = [self._extract_metadata(d, main_dir, client, sem) for d in subdirs]

            for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                try:
                    r = await coro
                    metadata[r.get("sub_dir")] = r.get("meta")
                except Exception as e:
                    print(f"Failed on {coro}: {e}")
            
        return metadata


