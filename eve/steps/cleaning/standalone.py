"""
Standalone utilities for running cleaning components independently.

This module provides functionality to run individual cleaning processors
or the entire cleaning pipeline as standalone operations, useful for
testing and debugging specific components.
"""

import asyncio
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any

from eve.model.document import Document
from eve.steps.cleaning.processors import (
    OCRProcessor,
    DuplicateRemovalProcessor,
    NougatProcessor,
    RuleBasedProcessor,
    LaTeXProcessor,
)
from eve.logging import get_logger

logger = get_logger(__name__)


class StandaloneCleaningRunner:
    """Runner for executing cleaning components independently."""
    
    def __init__(self, debug: bool = False):
        """Initialize the standalone runner.
        
        Args:
            debug: Enable debug output
        """
        self.debug = debug
        self.processors = {
            'ocr': OCRProcessor,
            'duplicates': DuplicateRemovalProcessor,
            'nougat': NougatProcessor,
            'rules': RuleBasedProcessor,
            'latex': LaTeXProcessor,
        }
    
    async def run_processor(
        self, 
        processor_name: str, 
        input_files: List[Path],
        processor_config: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Run a specific processor on input files.
        
        Args:
            processor_name: Name of the processor to run
            input_files: List of input file paths
            processor_config: Optional configuration for the processor
            
        Returns:
            List of processed documents
        """
        if processor_name not in self.processors:
            raise ValueError(f"Unknown processor: {processor_name}. Available: {list(self.processors.keys())}")
        
        processor_class = self.processors[processor_name]
        config = processor_config or {}
        config['debug'] = self.debug
        
        # Create processor instance
        if processor_name == 'duplicates':
            processor = processor_class(
                threshold=config.get('threshold', 0.99),
                min_words=config.get('min_words', 2),
                debug=self.debug
            )
        elif processor_name == 'latex':
            processor = processor_class(
                debug=self.debug,
                api_key=config.get('api_key'),
                model=config.get('model', 'anthropic/claude-3-haiku')
            )
        else:
            processor = processor_class(debug=self.debug)
        
        # Load documents from files
        documents = []
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document.from_path_and_content(file_path, content))
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                continue
        
        if not documents:
            logger.warning("No documents loaded successfully")
            return []
        
        # Process documents
        results = []
        for document in documents:
            try:
                processed = await processor.process(document)
                results.append(processed)
                logger.info(f"Processed {document.filename} with {processor_name}")
            except Exception as e:
                logger.error(f"Failed to process {document.filename} with {processor_name}: {str(e)}")
                results.append(document)  # Return original document on error
        
        return results
    
    async def run_all_processors(
        self, 
        input_files: List[Path],
        processor_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Document]:
        """Run all processors in sequence on input files.
        
        Args:
            input_files: List of input file paths
            processor_configs: Optional configurations for each processor
            
        Returns:
            List of processed documents
        """
        configs = processor_configs or {}
        
        # Load documents from files
        documents = []
        for file_path in input_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append(Document.from_path_and_content(file_path, content))
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {str(e)}")
                continue
        
        if not documents:
            logger.warning("No documents loaded successfully")
            return []
        
        # Process through all processors in order
        processor_order = ['ocr', 'duplicates', 'nougat', 'rules']
        
        # Add LaTeX processor if configuration is provided
        if 'latex' in configs and configs['latex'].get('enable', False):
            processor_order.append('latex')
        
        current_documents = documents
        
        for processor_name in processor_order:
            logger.info(f"Running {processor_name} processor on {len(current_documents)} documents")
            
            processor_class = self.processors[processor_name]
            config = configs.get(processor_name, {})
            config['debug'] = self.debug
            
            # Create processor instance
            if processor_name == 'duplicates':
                processor = processor_class(
                    threshold=config.get('threshold', 0.99),
                    min_words=config.get('min_words', 2),
                    debug=self.debug
                )
            elif processor_name == 'latex':
                processor = processor_class(
                    debug=self.debug,
                    api_key=config.get('api_key'),
                    model=config.get('model', 'anthropic/claude-3-haiku')
                )
            else:
                processor = processor_class(debug=self.debug)
            
            # Process all documents
            processed_documents = []
            for document in current_documents:
                try:
                    processed = await processor.process(document)
                    processed_documents.append(processed)
                except Exception as e:
                    logger.error(f"Failed to process {document.filename} with {processor_name}: {str(e)}")
                    processed_documents.append(document)  # Return original document on error
            
            current_documents = processed_documents
        
        return current_documents
    
    def save_results(self, documents: List[Document], output_dir: Path) -> None:
        """Save processed documents to output directory.
        
        Args:
            documents: List of processed documents
            output_dir: Output directory path
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for document in documents:
            output_path = output_dir / document.filename
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(document.content)
                logger.info(f"Saved processed document to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save {output_path}: {str(e)}")


async def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="Run cleaning components independently")
    parser.add_argument("input_files", nargs="+", type=Path, help="Input files to process")
    parser.add_argument("--processor", choices=['ocr', 'duplicates', 'nougat', 'rules', 'latex', 'all'], 
                       default='all', help="Processor to run")
    parser.add_argument("--output-dir", type=Path, default=Path("./cleaned_output"), 
                       help="Output directory for processed files")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--threshold", type=float, default=0.99, 
                       help="Similarity threshold for duplicate removal")
    parser.add_argument("--min-words", type=int, default=2, 
                       help="Minimum words for duplicate processing")
    parser.add_argument("--latex-api-key", type=str, help="OpenRouter API key for LaTeX correction")
    parser.add_argument("--latex-model", type=str, default="anthropic/claude-3-haiku", 
                       help="Model for LaTeX correction")
    
    args = parser.parse_args()
    
    # Validate input files
    valid_files = []
    for file_path in args.input_files:
        if file_path.exists() and file_path.is_file():
            valid_files.append(file_path)
        else:
            logger.warning(f"File not found or not a file: {file_path}")
    
    if not valid_files:
        logger.error("No valid input files provided")
        return
    
    # Create runner
    runner = StandaloneCleaningRunner(debug=args.debug)
    
    # Run processor(s)
    if args.processor == 'all':
        configs = {
            'duplicates': {
                'threshold': args.threshold,
                'min_words': args.min_words
            }
        }
        
        if args.latex_api_key:
            configs['latex'] = {
                'enable': True,
                'api_key': args.latex_api_key,
                'model': args.latex_model
            }
        
        results = await runner.run_all_processors(valid_files, configs)
    else:
        config = {}
        if args.processor == 'duplicates':
            config = {
                'threshold': args.threshold,
                'min_words': args.min_words
            }
        elif args.processor == 'latex':
            config = {
                'api_key': args.latex_api_key,
                'model': args.latex_model
            }
        
        results = await runner.run_processor(args.processor, valid_files, config)
    
    # Save results
    runner.save_results(results, args.output_dir)
    
    logger.info(f"Processing complete. {len(results)} documents saved to {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
