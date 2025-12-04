# Core Components

This section covers the core components of the EVE Pipeline that form the foundation of the data processing framework.

## Pipeline

The main pipeline orchestrator that coordinates all processing stages. This is where the files are first batched, converted to document objects, then passed to each pipeline stages.

```python
async def pipeline():
    logger = get_logger("pipeline")
    cfg = load_config("config.yaml")

    batch_size = cfg.batch_size

    logger.info("Starting pipeline execution")

    input_files = cfg.inputs.get_files()

    logger.info(f"Processing {len(input_files)} files with batch size {batch_size}")

    unique_file_formats = {find_format(f) for f in input_files}

    stages_with_extraction_dependency = {"dedup", "cleaning", "pii"}

    if 'md' not in unique_file_formats:
        user_stage_names = {stage["name"] for stage in cfg.stages}
        if not any(stage in user_stage_names for stage in stages_with_extraction_dependency):
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
    }

    batchable_steps = {"cleaning", "extraction", "pii", "metadata", "export"}
    
    has_dedup = any(stage["name"] == "duplication" for stage in cfg.stages)
    
    if has_dedup: 
        logger.info("Deduplication detected - collecting all documents before processing")
        all_documents = []
        async for batch in create_batches(input_files, batch_size):
            batch_docs = batch
            for stage in cfg.stages:
                step_name = stage["name"]
                if step_name == "duplication":
                    break  # stop here, accumulate all docs and run dedup in phase 2
                if step_name in batchable_steps and step_name in step_mapping:
                    step_config = stage.get("config", {})
                    step = step_mapping[step_name](config = step_config)
                    logger.info(f"Running step on batch: {step_name}")
                    batch_docs = await step(batch_docs)
            
            all_documents.extend(batch_docs)
        
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
        all_processed = []
        
        async for batch in create_batches(input_files, batch_size):
            batch_docs = batch
            logger.info(f"Processing batch of {len(batch_docs)} documents")
            
            for stage in cfg.stages:
                step_name = stage["name"]
                step_config = stage.get("config", {})
                if step_name in step_mapping:
                    step = step_mapping[step_name](config = step_config)
                    logger.info(f"Running step on batch: {step_name}")
                    batch_docs = await step(batch_docs)
                else:
                    logger.error(f"No implementation found for step: {step_name}")
            
            all_processed.extend(batch_docs)
        
        documents = all_processed
```

## Configuration

Configuration management for the pipeline using Pydantic models. These objects provide type-safe settings validation and management for all pipeline components.


```python
class PipelineConfig(BaseModel):
    batch_size: int = 20
    inputs: Inputs
    stages: list[dict[str, Any]]

    @validator("stages")
    def check_stages(cls, v):
        allowed = {"ingestion", "cleaning", "export", "duplication", "extraction", "pii", "metadata"}
        for stage in v:
            if stage["name"] not in allowed:
                raise ValueError(f"Unsupported stage: {stage['name']}. Allowed: {allowed}")
        return v
```


## Document Model

The unified document object that represents content and metadata throughout the pipeline. Documents are the core data structure that flow through the pipeline, containing both content and associated metadata.

```python
class Document:
    content: str
    file_path: Path
    file_format: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.file_path)

    def __eq__(self, other):
        return isinstance(other, Document) and self.file_path == other.file_path
    
    @property
    def filename(self) -> str:
        """Get the filename without path."""
        return self.file_path.name
    
    @property
    def extension(self) -> str:
        """Get the file extension."""
        return self.file_path.suffix.lstrip('.')
    
    @property
    def content_length(self) -> int:
        """Get the length of the content."""
        return len(self.content)
    
    def is_empty(self) -> bool:
        """Check if the document content is empty."""
        return not self.content.strip()
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add a metadata entry."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value with optional default."""
        return self.metadata.get(key, default)
    
    def update_content(self, new_content: str) -> None:
        """Update the document content and track the change in metadata."""
        old_length = self.content_length
        self.content = new_content
        new_length = self.content_length
        
        # Track content changes in metadata
        changes = self.metadata.get('content_changes', [])
        changes.append({
            'old_length': old_length,
            'new_length': new_length,
            'size_change': new_length - old_length
        })
        self.metadata['content_changes'] = changes
    
    @classmethod
    def from_path_and_content(cls, file_path: Path, content: str, **metadata) -> 'Document':
        """Create a Document from a file path and content string."""
        return cls(
            content=content,
            file_path=file_path,
            metadata=metadata
        )
    
    @classmethod
    def from_tuple(cls, path_content_tuple: tuple[Path, str], **metadata) -> 'Document':
        """Create a Document from a (Path, str) tuple for backwards compatibility."""
        file_path, content = path_content_tuple
        return cls.from_path_and_content(file_path, content, **metadata)
    
    def to_tuple(self) -> tuple[Path, str]:
        """Convert to (Path, str) tuple for backwards compatibility."""
        return (self.file_path, self.content)
    
    def __str__(self) -> str:
        """String representation showing filename and content length."""
        return f"Document({self.filename}, {self.file_format} format)"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"Document(file_path={self.file_path}, format={self.file_format}, metadata_keys={list(self.metadata.keys())})"
```

## Pipeline Step Base

Abstract base class that all pipeline stages must implement. All custom pipeline components should inherit from PipelineStep to ensure proper integration with the framework.

```python
class PipelineStep(ABC):
    """abstract base class for all pipeline steps."""

    def __init__(self, config: Any, name: Optional[str] = None):
        """initialize the pipeline step.

        Args:
            config: Configuration specific to the step.
            name: Optional name for the step (used for logging).
        """
        self.config = config
        self.debug = config.get("debug", False) if isinstance(config, dict) else False
        self.logger = get_logger(name or self.__class__.__name__)

    @abstractmethod
    async def execute(self, input_data: Any) -> Any: # TBD
        """Execute the pipeline step.

        Args:
            input_data: Input data to process.

        Returns:
            Processed data or result of the step.
        """
        pass

    async def __call__(self, input_data: Any) -> Any:
        """shortway of calling `execute` method.

        Args:
            input_data: Input data to process.

        Returns:
            Processed data or result of the step.
        """
        return await self.execute(input_data)

```