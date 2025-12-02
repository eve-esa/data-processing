# Core Components

This section covers the core components of the EVE Pipeline that form the foundation of the data processing framework.

## Pipeline

The main pipeline orchestrator that coordinates all processing stages.

```python
from eve.pipeline import Pipeline
```

The pipeline manages the execution flow through all configured stages, handling data transformation, error recovery, and progress tracking.

## Configuration

Configuration management for the pipeline using Pydantic models.

```python
from eve.config import PipelineConfig
```

Configuration objects provide type-safe settings validation and management for all pipeline components.

## Document Model

The unified document object that represents content and metadata throughout the pipeline.

```python
from eve.model.document import Document
```

Documents are the core data structure that flow through the pipeline, containing both content and associated metadata.

## Pipeline Step Base

Abstract base class that all pipeline stages must implement.

```python
from eve.base_step import PipelineStep
```

All custom pipeline components should inherit from PipelineStep to ensure proper integration with the framework.