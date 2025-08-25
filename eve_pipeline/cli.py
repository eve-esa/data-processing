"""Main CLI interface for the Eve data processing pipeline."""

import atexit
import contextlib
import json
import logging
import multiprocessing
import sys
from pathlib import Path
from typing import Optional

import click

from eve_pipeline.core.base import ProcessorStatus
from eve_pipeline.core.config import PipelineConfig
from eve_pipeline.core.pipeline import Pipeline

# Global reference to pipeline for cleanup
_pipeline_instance = None

def cleanup_on_exit():
    """Clean up resources on exit to prevent memory leaks."""
    global _pipeline_instance
    if _pipeline_instance:
        try:
            _pipeline_instance._cleanup_resources()
        except Exception as e:
            logger = logging.getLogger("eve_pipeline.cleanup")
            logger.warning(f"Error during exit cleanup: {e}")

        # Final PyTorch cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        # Force final garbage collection
        import gc
        gc.collect()

# Register cleanup function
atexit.register(cleanup_on_exit)


@click.group()
@click.version_option(version="0.1.0", prog_name="eve-pipeline")
def main():
    """Eve Data Processing Pipeline - A scalable, modular data processing pipeline."""
    # Set multiprocessing start method to 'spawn' on macOS to avoid trace trap errors
    if sys.platform == "darwin":  # macOS
        with contextlib.suppress(RuntimeError):
            multiprocessing.set_start_method('spawn', force=True)


@main.command()
@click.argument('input_path', type=str)
@click.option('--additional-files', '-f', multiple=True, help='Additional files to process (can be used multiple times)')
@click.option('--output', '-o', type=str, help='Output file or directory path (local or S3)')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--extraction/--no-extraction', default=True, help='Enable/disable extraction stage')
@click.option('--cleaning/--no-cleaning', default=True, help='Enable/disable cleaning stage')
@click.option('--pii-removal/--no-pii-removal', default=True, help='Enable/disable PII removal stage')
@click.option('--deduplication/--no-deduplication', default=False, help='Enable/disable deduplication stage (only useful for directory processing)')
@click.option('--latex-correction/--no-latex-correction', default=True, help='Enable/disable LaTeX correction stage')
@click.option('--num-processes', type=int, help='Number of parallel processes')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--save-results', type=click.Path(), help='Save processing results to JSON file')
@click.option('--aws-access-key-id', type=str, help='AWS access key ID (overrides config/env)')
@click.option('--aws-secret-access-key', type=str, help='AWS secret access key (overrides config/env)')
@click.option('--aws-region', type=str, help='AWS region (overrides config/env)')
@click.option('--aws-session-token', type=str, help='AWS session token (overrides config/env)')
def process(
    input_path: str,
    additional_files: tuple[str, ...],
    output: Optional[str],
    config: Optional[str],
    extraction: bool,
    cleaning: bool,
    pii_removal: bool,
    deduplication: bool,
    latex_correction: bool,
    num_processes: Optional[int],
    debug: bool,
    save_results: Optional[str],
    aws_access_key_id: Optional[str],
    aws_secret_access_key: Optional[str],
    aws_region: Optional[str],
    aws_session_token: Optional[str],
):
    """Process files through the complete pipeline."""

    # Early input validation before loading any modules
    import os
    from pathlib import Path
    
    # Collect all files to process
    all_files = [input_path] + list(additional_files)
    supported_extensions = {'.pdf', '.xml', '.html', '.htm', '.txt', '.text', '.csv', '.tsv', '.md', '.markdown'}
    
    # Validate all files
    for file_path in all_files:
        if file_path.startswith('s3://'):
            # For S3 paths, we'll defer validation until storage is initialized
            continue
        
        # For local paths, validate immediately
        path_obj = Path(file_path)
        if not path_obj.exists():
            click.echo(f"Error: Input file does not exist: {file_path}", err=True)
            sys.exit(1)
        
        if not path_obj.is_file():
            click.echo(f"Error: Path is not a file: {file_path}", err=True)
            sys.exit(1)
        
        # Check if it's a supported file type
        if path_obj.suffix.lower() not in supported_extensions:
            click.echo(f"Error: Unsupported file type '{path_obj.suffix}' for file '{file_path}'. Supported types: {', '.join(sorted(supported_extensions))}", err=True)
            sys.exit(1)

    # Load configuration
    pipeline_config = PipelineConfig.from_file(config) if config else PipelineConfig()

    # Override configuration with CLI options
    pipeline_config.extraction.enabled = extraction
    pipeline_config.cleaning.enabled = cleaning
    pipeline_config.pii_removal.enabled = pii_removal
    pipeline_config.deduplication.enabled = deduplication
    pipeline_config.latex_correction.enabled = latex_correction
    pipeline_config.debug = debug

    if num_processes:
        pipeline_config.num_processes = num_processes

    # Check for environment variable to disable multiprocessing (useful for debugging on macOS)
    import os
    if os.getenv('EVE_DISABLE_MULTIPROCESSING', '').lower() in ('1', 'true', 'yes'):
        pipeline_config.num_processes = 1

    if aws_access_key_id:
        pipeline_config.storage.aws_access_key_id = aws_access_key_id
    if aws_secret_access_key:
        pipeline_config.storage.aws_secret_access_key = aws_secret_access_key
    if aws_region:
        pipeline_config.storage.aws_region = aws_region
    if aws_session_token:
        pipeline_config.storage.aws_session_token = aws_session_token

    # Configure logging to show errors in terminal
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
    )

    # Initialize pipeline
    try:
        pipeline = Pipeline(config=pipeline_config)
        pipeline.update_stage_configuration()
        # Store global reference for cleanup
        global _pipeline_instance
        _pipeline_instance = pipeline
    except Exception as e:
        click.echo(f"Error: Failed to initialize pipeline: {e}", err=True)
        sys.exit(1)

    from eve_pipeline.storage.factory import StorageFactory

    # Get storage interface for processing
    input_storage = StorageFactory.get_storage_for_path(input_path, **pipeline_config.storage.to_storage_kwargs())
    
    # Final validation for S3 paths that couldn't be validated earlier
    if input_path.startswith('s3://') and not input_storage.exists(input_path):
        click.echo(f"Error: S3 path does not exist: {input_path}", err=True)
        sys.exit(1)

    # Process input
    if len(all_files) == 1 and input_storage.is_file(input_path):
        # Single file processing
        click.echo(f"Processing file: {input_path}")
        
        # Warn if deduplication is enabled for single file
        if deduplication:
            click.echo("⚠ Warning: Deduplication is enabled for single file processing. This stage is not useful for single files.", err=True)

        result = pipeline.process_file(input_path, output)

        if result.is_success:
            click.echo("✓ Processing completed successfully")
            if result.output_path:
                click.echo(f"Output saved to: {result.output_path}")
            click.echo(f"Processing time: {result.processing_time:.2f}s")

            user_meta = result.get_user_metadata()
            if user_meta:
                if "stages_completed" in user_meta:
                    click.echo(f"Stages completed: {user_meta['stages_completed']}")
                if "content_length" in user_meta:
                    click.echo(f"Content length: {user_meta['content_length']:,} characters")
                if debug and result.metadata:
                    click.echo(f"Debug info: {len(result.metadata)} metadata fields available")

        elif result.is_skipped:
            click.echo(f"⚠ File skipped: {result.metadata.get('skip_reason', 'Unknown reason')}")

        else:
            click.echo(f"✗ Processing failed: {result.error_message}", err=True)
            
            # Show more detailed error information if available
            if result.metadata and "processing_steps" in result.metadata:
                failed_steps = [step for step in result.metadata["processing_steps"] if step.get("status") == "FAILED"]
                if failed_steps:
                    click.echo("\nDetailed error information:", err=True)
                    for step in failed_steps:
                        click.echo(f"  Stage: {step.get('stage', 'Unknown')}", err=True)
                        if "error" in step:
                            click.echo(f"  Error: {step['error']}", err=True)
                        if "metadata" in step and step["metadata"]:
                            click.echo(f"  Details: {step['metadata']}", err=True)
            
            if debug:
                click.echo(f"\nFull debug metadata: {result.get_debug_metadata()}", err=True)
            
            sys.exit(1)

        final_result = {
            "type": "single_file",
            "input": input_path,
            "output": str(result.output_path) if result.output_path else None,
            "status": result.status.value,
            "processing_time": result.processing_time,
            "metadata": result.get_user_metadata(),
        }

    elif len(all_files) > 1:
        # Multiple files processing
        click.echo(f"Processing {len(all_files)} files:")
        for i, file_path in enumerate(all_files, 1):
            click.echo(f"  {i}. {file_path}")
        
        # Warn if deduplication is enabled for multiple files
        if deduplication:
            click.echo("ℹ Info: Deduplication is enabled - duplicate files will be skipped.")
        
        # Process all files
        import time
        start_time = time.time()
        all_results = []
        successful = 0
        failed = 0
        skipped = 0
        
        for file_path in all_files:
            click.echo(f"\nProcessing: {file_path}")
            
            # Create output path for this file if output is a directory
            if output and Path(output).is_dir():
                input_name = Path(file_path).stem
                file_output = str(Path(output) / f"{input_name}_processed.md")
            else:
                file_output = None
                
            result = pipeline.process_file(file_path, file_output)
            all_results.append(result)
            
            if result.is_success:
                successful += 1
                click.echo(f"  ✓ Completed in {result.processing_time:.2f}s")
                if result.output_path:
                    click.echo(f"  → Output: {result.output_path}")
            elif result.is_skipped:
                skipped += 1
                click.echo(f"  ⚠ Skipped: {result.metadata.get('skip_reason', 'Unknown reason')}")
            else:
                failed += 1
                click.echo(f"  ✗ Failed: {result.error_message}")
        
        total_time = time.time() - start_time
        
        # Summary
        click.echo(f"\n📊 Processing Summary:")
        click.echo(f"  Total files: {len(all_files)}")
        click.echo(f"  Successful: {successful}")
        click.echo(f"  Failed: {failed}")
        click.echo(f"  Skipped: {skipped}")
        click.echo(f"  Success rate: {(successful / len(all_files) * 100):.1f}%")
        click.echo(f"  Total time: {total_time:.2f}s")
        
        # Show failed files details
        failed_results = [r for r in all_results if r.is_failed]
        if failed_results:
            click.echo(f"\n❌ Failed files:")
            for result in failed_results[:5]:  # Show first 5
                click.echo(f"  {result.input_path}: {result.error_message}")
            if len(failed_results) > 5:
                click.echo(f"  ... and {len(failed_results) - 5} more")
        
        final_result = {
            "type": "multiple_files",
            "files": [str(f) for f in all_files],
            "total_files": len(all_files),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "success_rate": successful / len(all_files) * 100,
            "total_processing_time": total_time,
            "results": [{"input": str(r.input_path), "status": r.status.value, "output": str(r.output_path) if r.output_path else None, "error": r.error_message, "metadata": r.get_user_metadata()} for r in all_results]
        }

    elif input_storage.is_dir(input_path):
        # Directory processing
        click.echo(f"Processing directory: {input_path}")

        # Show a progress indicator
        import sys
        if not debug and sys.stdout.isatty():
            click.echo("Processing files... (use --debug for detailed progress)")

        results = pipeline.process_directory(input_path, output)

        if results["success"]:
            stats = results.get("statistics", {})
            click.echo("✓ Processing completed")
            click.echo(f"Total files: {stats.get('total_files', 0)}")
            click.echo(f"Successful: {stats.get('successful', 0)}")
            click.echo(f"Failed: {stats.get('failed', 0)}")
            click.echo(f"Skipped: {stats.get('skipped', 0)}")
            click.echo(f"Success rate: {stats.get('success_rate', 0):.1f}%")
            click.echo(f"Total processing time: {stats.get('total_processing_time', 0):.2f}s")

            if stats.get('failed', 0) > 0:
                click.echo("⚠ Some files failed to process. Check logs for details.")

                # Show details of failed files
                failed_results = [r for r in results.get("results", []) if r.status == ProcessorStatus.FAILED]
                if failed_results and len(failed_results) <= 10:  # Don't spam if too many failures
                    click.echo("\n📋 Failed files details:")
                    for i, result in enumerate(failed_results[:10], 1):
                        click.echo(f"  {i}. {result.input_path}")
                        click.echo(f"     Error: {result.error_message}")
                elif len(failed_results) > 10:
                    click.echo(f"\n📋 First 10 failed files (out of {len(failed_results)}):")
                    for i, result in enumerate(failed_results[:10], 1):
                        click.echo(f"  {i}. {result.input_path}")
                        click.echo(f"     Error: {result.error_message}")
                    click.echo(f"     ... and {len(failed_results) - 10} more failures")
                    click.echo("     Use --save-results to see all details")
        else:
            click.echo(f"✗ Directory processing failed: {results.get('error_message', 'Unknown error')}", err=True)
            
            # Show additional debug information if available
            if debug and results.get("statistics"):
                stats = results["statistics"]
                click.echo(f"\nPartial statistics:", err=True)
                click.echo(f"  Files found: {stats.get('total_files', 0)}", err=True)
                click.echo(f"  Successfully processed: {stats.get('successful', 0)}", err=True)
                click.echo(f"  Failed: {stats.get('failed', 0)}", err=True)
            
            sys.exit(1)

        final_result = results

    else:
        click.echo("Error: Input path must be a file or directory", err=True)
        sys.exit(1)

    # Save results if requested
    if save_results:
        try:
            with open(save_results, 'w') as f:
                json.dump(final_result, f, indent=2, default=str)
            click.echo(f"Results saved to: {save_results}")
        except Exception as e:
            click.echo(f"Warning: Failed to save results: {e}", err=True)


@main.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--file-pattern', default='*.md', help='File pattern to match')
@click.option('--save-duplicates', type=click.Path(), help='Save duplicate groups to file')
@click.option('--exact/--no-exact', default=True, help='Enable exact deduplication')
@click.option('--lsh/--no-lsh', default=True, help='Enable LSH-based near-duplicate detection')
@click.option('--threshold', type=float, default=0.8, help='LSH similarity threshold')
def deduplicate(
    input_dir: str,
    file_pattern: str,
    save_duplicates: Optional[str],
    exact: bool,
    lsh: bool,
    threshold: float,
):
    """Find and report duplicates in a directory."""

    from eve_pipeline.core.config import DeduplicationConfig
    from eve_pipeline.deduplication.pipeline import DeduplicationPipeline

    config = DeduplicationConfig(
        exact_deduplication=exact,
        lsh_deduplication=lsh,
        lsh_threshold=threshold,
    )

    pipeline = DeduplicationPipeline(config=config)

    click.echo(f"Analyzing duplicates in: {input_dir}")
    click.echo(f"File pattern: {file_pattern}")

    results = pipeline.process_directory(input_dir, file_pattern)

    if results["success"]:
        stats = results["statistics"]
        click.echo("✓ Analysis completed")
        click.echo(f"Total files: {stats['total_files']}")
        click.echo(f"Exact duplicates: {stats['exact_duplicates']} files in {stats['exact_duplicate_groups']} groups")
        click.echo(f"Near duplicates: {stats['near_duplicates']} files in {stats['near_duplicate_groups']} groups")
        click.echo(f"Unique files: {stats['unique_files']}")
        click.echo(f"Deduplication rate: {stats['deduplication_rate']:.1f}%")

        if save_duplicates:
            try:
                with open(save_duplicates, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                click.echo(f"Duplicate analysis saved to: {save_duplicates}")
            except Exception as e:
                click.echo(f"Warning: Failed to save results: {e}", err=True)
    else:
        click.echo(f"✗ Analysis failed: {results.get('error_message', 'Unknown error')}", err=True)
        sys.exit(1)


@main.command()
@click.option('--output', '-o', type=click.Path(), default='pipeline_config.yaml', help='Output configuration file')
@click.option('--format', 'config_format', type=click.Choice(['yaml', 'json']), default='yaml', help='Configuration format')
def init_config(output: str, config_format: str):
    """Initialize a default configuration file."""

    config = PipelineConfig()

    output_path = Path(output)
    if config_format == 'yaml':
        if not output_path.suffix:
            output_path = output_path.with_suffix('.yaml')
    else:
        if not output_path.suffix:
            output_path = output_path.with_suffix('.json')

    try:
        config.save(output_path)
        click.echo(f"✓ Configuration file created: {output_path}")
        click.echo("Edit this file to customize your pipeline settings.")
    except Exception as e:
        click.echo(f"Error: Failed to create configuration file: {e}", err=True)
        sys.exit(1)


@main.command()
@click.option('--port', type=int, default=8000, help='Port to run server on')
@click.option('--workers', type=int, default=1, help='Number of workers')
@click.option('--pii-only', is_flag=True, help='Run PII removal server only')
def serve(port: int, workers: int, pii_only: bool):
    """Start a processing server."""

    if pii_only:
        import sys

        from eve_pipeline.pii_removal.server import main as pii_main

        # Set up arguments for PII server
        sys.argv = ['pii_server', '--port', str(port), '--workers', str(workers)]
        pii_main()
    else:
        click.echo("Full pipeline server not yet implemented. Use --pii-only for PII removal server.")
        sys.exit(1)


if __name__ == "__main__":
    main()
