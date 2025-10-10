import argparse
import asyncio
import time

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
from eve.utils import find_format

async def pipeline():
    logger = get_logger("pipeline")
    cfg = load_config("config.yaml")

    logger.info("Starting pipeline execution")

    start_time = time.perf_counter()
    input_files = cfg.inputs.get_files()

    documents = []
    unique_file_formats = set()
    for file_path in input_files:
        doc = Document(
            file_path = file_path,
            content = "",
            file_format = find_format(file_path),
        )
        unique_file_formats.add(doc.file_format)

        documents.append(doc)

    stages_with_extraction_dependency = {"dedup", "cleaning", "pii"}

    # enable extraction only if needed
    if 'md' not in unique_file_formats:
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
        "pii_filter": PiiFilterStep
    }

    for stage in cfg.stages:
        step_name = stage["name"]
        step_config = stage.get("config", {})
        if step_name in step_mapping:
            step = step_mapping[step_name](config = step_config)
            logger.info(f"Running step: {step_name}")
            documents = await step(documents)
        else:
            logger.error(f"No implementation found for step: {step_name}")
    
    end_time = time.perf_counter()  # end timer
    elapsed_time = end_time - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

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
