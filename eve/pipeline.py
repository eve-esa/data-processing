import asyncio

from eve.config import load_config
from eve.logging import get_logger
from eve.steps.dedup.dedup_step import DuplicationStep
from eve.steps.extraction.extract_step import ExtractionStep
from eve.steps.export.export_step import ExportStep
from eve.steps.cleaning.cleaning_step import CleaningStep

async def pipeline():
    logger = get_logger("pipeline")
    cfg = load_config("config.yaml")

    logger.info("Starting pipeline execution")

    logger.info(f"Stages: {[stage['name'] for stage in cfg.stages]}")
    logger.info("Files to process:")
    input_files = cfg.inputs.get_files()

    data = input_files
            
    step_mapping = {
        "cleaning": CleaningStep,
        "export": ExportStep,
        "duplication": DuplicationStep,
        "extraction": ExtractionStep,
    }

    for stage in cfg.stages:
        step_name = stage["name"]
        step_config = stage.get("config", {})
        if step_name in step_mapping:
            step = step_mapping[step_name](config = step_config)
            logger.info(f"Running step: {step_name}")
            data = await step(data)
        else:
            logger.error(f"No implementation found for step: {step_name}")

def main():
    """entry point for the pipeline"""
    return asyncio.run(pipeline())

if __name__ == "__main__":
    main()
