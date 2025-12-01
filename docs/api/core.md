# Core Components

This section covers the core components of the EVE Pipeline that form the foundation of the data processing framework.

## Pipeline

The main pipeline orchestrator that coordinates all processing stages.

::: eve.pipeline.pipeline
    :members:
    :show-inheritance:

## Configuration

Configuration management for the pipeline using Pydantic models.

::: eve.config
    :members:
    :show-inheritance:

## Document Model

The unified document object that represents content and metadata throughout the pipeline.

::: eve.model.document
    :members:
    :show-inheritance:

## Pipeline Step Base

Abstract base class that all pipeline stages must implement.

::: eve.base_step
    :members:
    :show-inheritance: