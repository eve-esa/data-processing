import asyncio

from eve.config import load_config
from eve.logging import get_logger
from eve.model.document import Document
from eve.steps.dedup.dedup_step import DuplicationStep
from eve.steps.extraction.extract_step import ExtractionStep
from eve.steps.export.export_step import ExportStep
from eve.steps.cleaning.cleaning_step import CleaningStep
from eve.steps.pii.pii_step import PiiStep
from eve.utils import find_format

async def pipeline():
    logger = get_logger("pipeline")
    cfg = load_config("config.yaml")

    logger.info("Starting pipeline execution")
    # logger.info("Files to process:")
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

    if len(unique_file_formats) >= 1 and 'md' not in unique_file_formats:
        if not any(stage["name"] == "extraction" for stage in cfg.stages):
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

def main():
    """entry point for the pipeline"""
    return asyncio.run(pipeline())

if __name__ == "__main__":
    main()
