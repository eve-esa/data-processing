import argparse
import asyncio
import time
import json
from typing import List, AsyncIterator
from tqdm import tqdm

from eve.config import load_config
from eve.logging import get_logger
from eve.model.document import Document
from eve.steps.chunking.chunker_step import ChunkerStep
from eve.steps.dedup.dedup_step import DuplicationStep
from eve.steps.extraction.extract_step import ExtractionStep
from eve.steps.export.export_step import ExportStep
from eve.steps.cleaning.cleaning_step import CleaningStep
from eve.steps.filters.perplexity import PerplexityFilterStep
from eve.steps.pii.pii_step import PiiStep
from eve.steps.metadata.metadata_step import MetadataStep
from eve.steps.filters.pii_filter import PiiFilterStep
from eve.steps.filters.length_filter import LengthFilterStep
from eve.steps.filters.newline_filter import NewLineFilterStep
from eve.steps.filters.reference_filter import ReferenceFilterStep
from eve.steps.qdrant.qdrant_step import QdrantUploadStep
from eve.checkpoint import CheckpointManager
from eve.utils import find_format
from pathlib import Path

def count_total_documents(input_files: List, batch_size: int) -> int:
    """Count total number of documents to process (for progress bar).

    Args:
        input_files: List of Path objects
        batch_size: Batch size for processing

    Returns:
        Total number of documents
    """
    total = 0
    for file_path in input_files:
        file_format = find_format(file_path)
        if file_format == "jsonl":
            # Count lines in JSONL file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():  # Non-empty line
                            total += 1
            except Exception:
                pass
        else:
            # Regular file counts as 1 document
            total += 1
    return total

async def create_batches(input_files: List, batch_size: int, checkpoint_manager=None) -> AsyncIterator[List[Document]]:
    """Create batches of Document objects from input files.

    For regular files: Creates one Document per file and batches them.
    For JSONL files: Reads documents from JSONL and batches them.

    Args:
        input_files: List of Path objects pointing to input files
        batch_size: Number of documents per batch
        checkpoint_manager: Optional CheckpointManager for resume functionality

    Yields:
        Batches of Document objects
    """
    batch = []
    logger = get_logger("pipeline.batching")

    for file_path in input_files:
        file_format = find_format(file_path)

        # Handle JSONL files specially - read and batch their contents
        if file_format == "jsonl":
            logger.info(f"Reading JSONL file for batching: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            json_doc = json.loads(line.strip())
                            if "content" not in json_doc:
                                logger.warning(f"No content found in {file_path} line {line_num}")
                                continue

                            # Create Document from JSONL line
                            doc = Document(
                                file_path=file_path,
                                content=json_doc["content"],
                                metadata=json_doc.get("metadata", {}),
                                embedding=json_doc.get("embedding", None),
                                pipeline_metadata=json_doc.get("pipeline_metadata", {}),
                                file_format="md",  # JSONL documents are treated as markdown
                            )

                            # Skip if already processed (resume mode)
                            if checkpoint_manager and checkpoint_manager.is_processed(doc):
                                continue

                            batch.append(doc)

                            # Yield batch when it reaches batch_size
                            if len(batch) >= batch_size:
                                yield batch
                                batch = []
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error in {file_path} line {line_num}: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error reading JSONL file {file_path}: {e}")
                continue
        else:
            # Handle regular files - create placeholder Document
            doc = Document(
                file_path=file_path,
                content="",
                file_format=file_format,
            )
            batch.append(doc)

            if len(batch) >= batch_size:
                yield batch
                batch = []

    # Yield final batch if any documents remain
    if batch:
        yield batch

async def pipeline():
    logger = get_logger("pipeline")
    cfg = load_config("config.yaml")

    batch_size = cfg.batch_size

    logger.info("Starting pipeline execution")

    start_time = time.perf_counter()
    input_files = cfg.inputs.get_files()

    logger.info(f"Processing {len(input_files)} files with batch size {batch_size}")

    unique_file_formats = {find_format(f) for f in input_files}

    stages_with_extraction_dependency = {"dedup", "cleaning", "pii"}

    # enable extraction only if needed
    # Skip auto-adding extraction if all files are JSONL (handled in create_batches)
    if 'md' not in unique_file_formats and unique_file_formats != {'jsonl'}:
        user_stage_names = {stage["name"] for stage in cfg.stages}
        if not any(stage in user_stage_names for stage in stages_with_extraction_dependency):
            # no dependency stage, skip extraction
            pass
        else:
            if "extraction" not in user_stage_names:
                cfg.stages.insert(0, {"name": "extraction"})

    # enable export by default
    if not any(stage["name"] == "export" for stage in cfg.stages):
        cfg.stages.append({"name": "export"})

    
    logger.info(f"Stages: {[stage['name'] for stage in cfg.stages]}")

    step_mapping = {
        "cleaning": CleaningStep,
        "export": ExportStep,
        "duplication": DuplicationStep,
        "extraction": ExtractionStep,
        "pii": PiiStep,
        "metadata": MetadataStep,
        "chunker": ChunkerStep,
        "perplexity": PerplexityFilterStep,
        "pii_filter": PiiFilterStep,
        "length_filter": LengthFilterStep,
        "newline_filter": NewLineFilterStep,
        "reference_filter": ReferenceFilterStep,
        "qdrant_upload": QdrantUploadStep
    }

    batchable_steps = {"cleaning", "extraction", "pii", "metadata", "export"}

    # Count total documents for progress bar
    total_docs = count_total_documents(input_files, batch_size)
    logger.info(f"Total documents to process: {total_docs}")

    # Initialize checkpoint manager if resume is enabled in export step
    checkpoint_manager = None
    export_stage = next((stage for stage in cfg.stages if stage["name"] == "export"), None)
    if export_stage and export_stage.get("config", {}).get("resume", False):
        export_config = export_stage.get("config", {})
        output_dir = Path(export_config.get("output_dir", "./output"))
        checkpoint_manager = CheckpointManager(output_dir, resume=True)
        stats = checkpoint_manager.get_stats()
        logger.info(f"Resume mode enabled: {stats['processed_count']} documents already processed")
        logger.info(f"Checkpoint file: {stats['checkpoint_file']}")

    has_dedup = any(stage["name"] == "duplication" for stage in cfg.stages) #TO-DO - is there a way to do dedup with batching?

    if has_dedup: #handle batching seperately
        logger.info("Deduplication detected - collecting all documents before processing")
        all_documents = []

        # Create progress bar for batch processing
        with tqdm(total=total_docs, desc="Processing batches (pre-dedup)", unit="doc") as pbar:
            async for batch in create_batches(input_files, batch_size, checkpoint_manager):
                batch_docs = batch
                for stage in cfg.stages:
                    step_name = stage["name"]
                    if step_name == "duplication":
                        break  # stop here, accumulate all docs and run dedup in phase 2
                    if step_name in batchable_steps and step_name in step_mapping:
                        step_config = stage.get("config", {})
                        step = step_mapping[step_name](config = step_config)
                        batch_docs = await step(batch_docs)

                all_documents.extend(batch_docs)
                pbar.update(len(batch))

        documents = all_documents
        dedup_started = False
        for stage in cfg.stages:
            step_name = stage["name"]
            if step_name == "duplication":
                dedup_started = True

            if dedup_started:
                step_config = stage.get("config", {})
                if step_name in step_mapping:
                    step = step_mapping[step_name](config = step_config)
                    logger.info(f"Running step: {step_name}")
                    documents = await step(documents)
                else:
                    logger.error(f"No implementation found for step: {step_name}")
    else:
        logger.info("No deduplication - using streaming batch processing")
        total_processed = 0

        # Create progress bar for batch processing
        with tqdm(total=total_docs, desc="Processing batches", unit="doc") as pbar:
            async for batch in create_batches(input_files, batch_size, checkpoint_manager):
                batch_docs = batch

                for stage in cfg.stages:
                    step_name = stage["name"]
                    step_config = stage.get("config", {})
                    if step_name in step_mapping:
                        step = step_mapping[step_name](config = step_config)
                        batch_docs = await step(batch_docs)
                    else:
                        logger.error(f"No implementation found for step: {step_name}")

                # Don't accumulate documents - they're already exported
                # Just track count for reporting
                total_processed += len(batch_docs)
                pbar.update(len(batch))

                # Clear batch to free memory
                del batch_docs
                del batch

        documents = []  # Empty list since documents were exported inline
        logger.info(f"Total documents processed in streaming mode: {total_processed}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
    if documents:  # Only log count if documents list is populated (dedup mode)
        logger.info(f"Processed {len(documents)} documents successfully")

def main():
    """entry point for the pipeline"""
    return asyncio.run(pipeline())

def cli():
    parser = argparse.ArgumentParser(prog = "eve")
    subparsers = parser.add_subparsers(dest = "command")

    _ = subparsers.add_parser("run", help = "Run the Eve pipeline")

    args = parser.parse_args()

    if args.command == "run":
        main()
    else:
        parser.print_help()
