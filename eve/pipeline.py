from eve.base_step import PipelineStep
from eve.config import load_config
from eve.logging import get_logger
from eve.steps.dedup.dedup_step import DuplicationStep
from eve.steps.extraction.extract_step import ExtractionStep

class CleaningStep(PipelineStep):
    def execute(self, input_data: list) -> list:
        self.logger.info("Executing cleaning step")
        return input_data

class ExportStep(PipelineStep):
    def execute(self, input_data: list) -> list:
        self.logger.info(f"Executing export step to {self.output_dir}")
        return input_data


def main():
    logger = get_logger("pipeline")
    cfg = load_config("config.yaml")

    logger.info("Starting pipeline execution")

    logger.info(f"Stages: {[stage['name'] for stage in cfg.stages]}")
    logger.info(f"Output format: {cfg.output_format}")
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
            step = step_mapping[step_name](config = step_config, output_dir = cfg.output_directory)
            logger.info(f"Running step: {step_name}")
            data = step(data)
        else:
            logger.error(f"No implementation found for step: {step_name}")

if __name__ == "__main__":
    main()
