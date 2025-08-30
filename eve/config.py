from pathlib import Path
from typing import List, Union
import yaml
from pydantic import BaseModel, validator

class Inputs(BaseModel):
    mode: str = "file"  # file | directory
    path: Union[str, List[str]]
    
    def get_files(self) -> List[Path]:
        paths = [self.path] if isinstance(self.path, str) else self.path
        #print(paths)
        files = []
        
        for p in paths:
            p = Path(p)
            
            if p.is_file():
                files.append(p)
            elif p.is_dir():
                files.extend(p.glob("*"))        
        return files

class PipelineConfig(BaseModel):
    inputs: Inputs
    output_directory: Path
    output_format: str
    stages: List[str]
    
    @validator("output_format")
    def check_format(cls, v):
        allowed = {"md"} # add if missing
        if v not in allowed:
            raise ValueError(f"Unsupported output_format: {v}. Allowed: {allowed}")
        return v
    
    @validator("stages")
    def check_stages(cls, v):
        allowed = {"ingestion", "cleaning", "export"} # add as when we expand
        for stage in v:
            if stage not in allowed:
                raise ValueError(f"Unsupported stage: {stage}. Allowed: {allowed}")
        return v

def load_config(path: str) -> PipelineConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return PipelineConfig(**raw["pipeline"]) # unpack