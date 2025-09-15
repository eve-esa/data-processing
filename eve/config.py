from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel, validator


class Inputs(BaseModel):
    mode: str = "file"  # file | directory
    path: Union[str, list[str]]

    def get_files(self) -> list[Path]:
        paths = [self.path] if isinstance(self.path, str) else self.path
        files = []

        for p in paths:
            p = Path(p)

            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend([f for f in p.rglob("*") if f.is_file()]) # recursive search across multiple levels
        return files

class PipelineConfig(BaseModel):
    inputs: Inputs
    stages: list[dict[str, Any]]  # list of dict since we have stage name + stage configs

    @validator("stages")
    def check_stages(cls, v):
        allowed = {"ingestion", "cleaning", "export", "duplication", "extraction", "pii", "metadata"}
        for stage in v:
            if stage["name"] not in allowed:
                raise ValueError(f"Unsupported stage: {stage['name']}. Allowed: {allowed}")
        return v

def load_config(path: str) -> PipelineConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw["pipeline"])  # unpack
