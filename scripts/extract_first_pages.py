#!/usr/bin/env python3
"""Extract first pages as PDFs; defaults: /data/pdfs -> /data/pdf_first_pages.

    python extract_first_pages.py
"""

import argparse
import os
from pathlib import Path
import sys
import tempfile

import fitz  # PyMuPDF


def extract_first_page(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with fitz.open(source) as document:
            if document.needs_pass:
                raise ValueError("PDF requires a password")
            if not document.page_count:
                raise ValueError("PDF has no pages")
            with fitz.open() as first_page:
                first_page.insert_pdf(document, from_page=0, to_page=0)
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, suffix=".tmp", delete=False
                ) as handle:
                    temporary = Path(handle.name)
                first_page.save(temporary, garbage=3, deflate=True)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/data/pdfs"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/pdf_first_pages"))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs")
    args = parser.parse_args()
    source_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    if not source_root.is_dir():
        parser.error(f"Input directory does not exist: {source_root}")
    if output_root == source_root or source_root in output_root.parents:
        parser.error("Output directory must be outside the input directory")

    written = skipped = failed = 0
    for source in source_root.rglob("*"):
        if not source.is_file() or source.suffix.lower() != ".pdf":
            continue
        destination = output_root / source.relative_to(source_root)
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            extract_first_page(source, destination)
            written += 1
        except Exception as error:
            failed += 1
            print(f"Failed: {source}: {error}", file=sys.stderr, flush=True)
            if isinstance(error, OSError) and error.errno in (28, 122):
                print("Stopping: destination storage is full.", file=sys.stderr)
                break
        if (written + failed) % 100 == 0:
            print(f"Written: {written}; skipped: {skipped}; failed: {failed}", flush=True)

    print(f"Done. Written: {written}; skipped: {skipped}; failed: {failed}")
    print(f"Output: {output_root}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
