"""
Diagnostic tool for troubleshooting slow processing and S3 output issues.
"""

import logging
import time
import boto3
from typing import Optional, Dict, Any
import psutil

from eve_pipeline.core.config import PipelineConfig
from eve_pipeline.storage.factory import StorageFactory


class PipelineDiagnostics:
    """Comprehensive diagnostics for pipeline performance issues."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize diagnostics with optional config."""
        self.config = PipelineConfig.from_file(config_path) if config_path else PipelineConfig()
        self.logger = logging.getLogger(__name__)
        
        # Setup basic logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def run_full_diagnostic(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive diagnostic check."""
        print("🔍 Starting Pipeline Diagnostics...")
        print("=" * 60)
        
        results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_path': input_path,
            'output_path': output_path,
            'issues_found': [],
            'recommendations': [],
            'system_info': {},
            'pipeline_config': {},
            'storage_tests': {},
        }
        
        # 1. System Resource Check
        print("📊 Checking system resources...")
        results['system_info'] = self._check_system_resources()
        
        # 2. Pipeline Configuration Check
        print("⚙️ Checking pipeline configuration...")
        results['pipeline_config'] = self._check_pipeline_config()
        
        # 3. Storage Connectivity Tests
        print("🗄️ Testing storage connectivity...")
        results['storage_tests'] = self._test_storage_connectivity(input_path, output_path)
        
        # 4. Input Path Analysis
        print("📁 Analyzing input path...")
        input_analysis = self._analyze_input_path(input_path)
        results['input_analysis'] = input_analysis
        
        # 5. Batch Size Analysis
        print("📦 Analyzing batch configuration...")
        results['batch_analysis'] = self._analyze_batch_configuration()
        
        # 6. Performance Bottleneck Detection
        print("🐌 Detecting performance bottlenecks...")
        results['bottlenecks'] = self._detect_bottlenecks()
        
        # Generate final report
        self._generate_report(results)
        
        return results
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        try:
            memory = psutil.virtual_memory()
            cpu_count = psutil.cpu_count()
            disk = psutil.disk_usage('/')
            
            resources = {
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent_used': memory.percent,
                'cpu_count': cpu_count,
                'cpu_percent': psutil.cpu_percent(interval=1),
                'disk_free_gb': disk.free / (1024**3),
                'disk_percent_used': (disk.used / disk.total) * 100,
            }
            
            # Check for resource constraints
            if memory.percent > 80:
                print("⚠️  WARNING: High memory usage detected!")
            if resources['cpu_percent'] > 90:
                print("⚠️  WARNING: High CPU usage detected!")
            if resources['disk_percent_used'] > 90:
                print("⚠️  WARNING: Low disk space!")
            
            print(f"   Memory: {resources['memory_available_gb']:.1f}GB available ({memory.percent:.1f}% used)")
            print(f"   CPU: {cpu_count} cores, {resources['cpu_percent']:.1f}% utilization")
            print(f"   Disk: {resources['disk_free_gb']:.1f}GB free")
            
            return resources
            
        except Exception as e:
            return {'error': str(e)}
    
    def _check_pipeline_config(self) -> Dict[str, Any]:
        """Check pipeline configuration for common issues."""
        config_info = {
            'enabled_stages': self.config.enabled_stages,
            'num_processes': self.config.num_processes,
            'debug_mode': self.config.debug,
            'storage_config': {
                'aws_region': self.config.storage.aws_region,
                'aws_access_key_id': bool(self.config.storage.aws_access_key_id),
                'aws_secret_access_key': bool(self.config.storage.aws_secret_access_key),
            }
        }
        
        print(f"   Enabled stages: {config_info['enabled_stages']}")
        print(f"   Parallel processes: {config_info['num_processes']}")
        print(f"   Debug mode: {config_info['debug_mode']}")
        
        # Check for performance issues
        if len(config_info['enabled_stages']) > 3:
            print("⚠️  INFO: Many pipeline stages enabled - consider disabling unused stages")
        
        if config_info['num_processes'] == 1:
            print("⚠️  INFO: Single-process mode - consider enabling parallel processing")
        
        return config_info
    
    def _test_storage_connectivity(self, input_path: str, output_path: Optional[str]) -> Dict[str, Any]:
        """Test storage connectivity and permissions."""
        tests = {
            'input_storage': {'accessible': False, 'type': 'unknown'},
            'output_storage': {'accessible': False, 'type': 'unknown'},
            'aws_credentials': {'valid': False},
        }
        
        # Test input storage
        try:
            input_storage = StorageFactory.get_storage_for_path(input_path, **self.config.storage.to_storage_kwargs())
            tests['input_storage']['type'] = 's3' if input_path.startswith('s3://') else 'local'
            tests['input_storage']['accessible'] = input_storage.exists(input_path)
            
            if tests['input_storage']['accessible']:
                print(f"✅ Input storage accessible: {input_path}")
            else:
                print(f"❌ Input storage not accessible: {input_path}")
                
        except Exception as e:
            tests['input_storage']['error'] = str(e)
            print(f"❌ Input storage error: {e}")
        
        # Test output storage
        if output_path:
            try:
                output_storage = StorageFactory.get_storage_for_path(output_path, **self.config.storage.to_storage_kwargs())
                tests['output_storage']['type'] = 's3' if output_path.startswith('s3://') else 'local'
                
                # Try to write a test file
                test_content = "test"
                test_path = f"{output_path.rstrip('/')}/diagnostic_test.txt"
                output_storage.write_text(test_path, test_content)
                tests['output_storage']['accessible'] = True
                
                # Clean up test file
                try:
                    if output_storage.exists(test_path):
                        # Note: No delete method in StorageBase, but write test succeeded
                        pass
                except:
                    pass
                
                print(f"✅ Output storage accessible: {output_path}")
                
            except Exception as e:
                tests['output_storage']['error'] = str(e)
                print(f"❌ Output storage error: {e}")
        
        # Test AWS credentials
        if input_path.startswith('s3://') or (output_path and output_path.startswith('s3://')):
            try:
                session = boto3.Session(
                    aws_access_key_id=self.config.storage.aws_access_key_id,
                    aws_secret_access_key=self.config.storage.aws_secret_access_key,
                    region_name=self.config.storage.aws_region
                )
                s3_client = session.client('s3')
                s3_client.list_buckets()
                tests['aws_credentials']['valid'] = True
                print("✅ AWS credentials valid")
                
            except Exception as e:
                tests['aws_credentials']['error'] = str(e)
                print(f"❌ AWS credentials error: {e}")
        
        return tests
    
    def _analyze_input_path(self, input_path: str) -> Dict[str, Any]:
        """Analyze input path characteristics."""
        analysis = {
            'path_type': 's3' if input_path.startswith('s3://') else 'local',
            'is_directory': False,
            'file_count': 0,
            'total_size_gb': 0,
            'sample_files': [],
        }
        
        try:
            storage = StorageFactory.get_storage_for_path(input_path, **self.config.storage.to_storage_kwargs())
            
            if storage.is_dir(input_path):
                analysis['is_directory'] = True
                files = storage.list_files(input_path, "*.md")
                analysis['file_count'] = len(files)
                analysis['sample_files'] = files[:5]  # First 5 files as sample
                
                print(f"   Input type: Directory with {analysis['file_count']} files")
                if analysis['file_count'] == 0:
                    print("❌ No files found in input directory!")
                elif analysis['file_count'] > 10000:
                    print(f"⚠️  Large dataset: {analysis['file_count']} files - consider batch processing")
                
            else:
                analysis['is_directory'] = False
                if storage.exists(input_path):
                    analysis['file_count'] = 1
                    print(f"   Input type: Single file")
                else:
                    print(f"❌ Input file does not exist: {input_path}")
            
        except Exception as e:
            analysis['error'] = str(e)
            print(f"❌ Error analyzing input path: {e}")
        
        return analysis
    
    def _analyze_batch_configuration(self) -> Dict[str, Any]:
        """Analyze batch processing configuration."""
        # Default batch sizes from our optimizations
        default_config = {
            'parallel_processes': self.config.num_processes,
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'recommended_batch_size': 10,
        }
        
        # Calculate recommended batch size based on available memory
        memory_gb = default_config['memory_available_gb']
        if memory_gb < 4:
            recommended_batch = 5
            print("⚠️  Low memory - small batch size recommended")
        elif memory_gb < 8:
            recommended_batch = 10
        elif memory_gb < 16:
            recommended_batch = 20
        else:
            recommended_batch = 50
        
        default_config['recommended_batch_size'] = recommended_batch
        
        print(f"   Available memory: {memory_gb:.1f}GB")
        print(f"   Recommended batch size: {recommended_batch}")
        print(f"   Parallel processes: {default_config['parallel_processes']}")
        
        return default_config
    
    def _detect_bottlenecks(self) -> Dict[str, Any]:
        """Detect common performance bottlenecks."""
        bottlenecks = {
            'identified_issues': [],
            'performance_suggestions': [],
        }
        
        # Check common bottleneck patterns
        memory = psutil.virtual_memory()
        
        if memory.percent > 80:
            bottlenecks['identified_issues'].append("High memory usage - may cause swapping")
            bottlenecks['performance_suggestions'].append("Reduce batch size or enable streaming")
        
        if self.config.num_processes == 1:
            bottlenecks['identified_issues'].append("Single-threaded processing")
            bottlenecks['performance_suggestions'].append("Enable parallel processing with --num-processes")
        
        # Check if all stages are enabled
        all_stages = ['extraction', 'cleaning', 'pii_removal', 'deduplication', 'latex_correction']
        enabled = self.config.enabled_stages
        if len(enabled) == len(all_stages):
            bottlenecks['identified_issues'].append("All pipeline stages enabled")
            bottlenecks['performance_suggestions'].append("Disable unused stages to improve speed")
        
        if not bottlenecks['identified_issues']:
            bottlenecks['identified_issues'].append("No obvious bottlenecks detected")
        
        for issue in bottlenecks['identified_issues']:
            print(f"   🐌 {issue}")
        
        return bottlenecks
    
    def _generate_report(self, results: Dict[str, Any]) -> None:
        """Generate final diagnostic report."""
        print("\n" + "=" * 60)
        print("📋 DIAGNOSTIC SUMMARY")
        print("=" * 60)
        
        # Quick fixes
        print("\n🚀 QUICK FIXES TO TRY:")
        
        if results['pipeline_config']['num_processes'] == 1:
            print("1. Enable parallel processing:")
            print("   eve-pipeline process <input> --output <output> --num-processes 4")
        
        if 'All pipeline stages enabled' in str(results.get('bottlenecks', {})):
            print("2. Disable unused stages for faster processing:")
            print("   eve-pipeline process <input> --output <output> --no-pii-removal --no-deduplication")
        
        if results['system_info'].get('memory_percent_used', 0) > 80:
            print("3. Reduce memory usage:")
            print("   - Close other applications")
            print("   - Use smaller batch sizes")
            print("   - Enable streaming processing")
        
        # Configuration issues
        storage_issues = []
        if not results['storage_tests']['input_storage'].get('accessible', False):
            storage_issues.append("❌ Cannot access input path")
        if not results['storage_tests']['output_storage'].get('accessible', False):
            storage_issues.append("❌ Cannot write to output path")
        if not results['storage_tests']['aws_credentials'].get('valid', True):
            storage_issues.append("❌ Invalid AWS credentials")
        
        if storage_issues:
            print("\n🔧 CONFIGURATION ISSUES:")
            for issue in storage_issues:
                print(f"   {issue}")
        
        # Performance recommendations
        print("\n⚡ PERFORMANCE RECOMMENDATIONS:")
        
        file_count = results['input_analysis'].get('file_count', 0)
        if file_count > 1000:
            print("   - Large dataset detected - use checkpointing")
            print("   - Consider processing in smaller batches")
        
        if results['system_info'].get('cpu_count', 1) > 2:
            cpu_count = results['system_info']['cpu_count']
            recommended_workers = min(cpu_count - 1, 8)
            print(f"   - Use parallel processing: --num-processes {recommended_workers}")
        
        print("   - Enable debug mode to see detailed progress: --debug")
        print("   - Monitor processing with: --save-results results.json")
        
        print("\n📞 If issues persist:")
        print("   1. Check logs for detailed error messages")
        print("   2. Verify AWS credentials and permissions")
        print("   3. Test with a small sample first")
        print("   4. Enable debug mode for verbose output")


def main():
    """CLI entry point for diagnostics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Performance Diagnostics")
    parser.add_argument("input_path", help="Input path to analyze")
    parser.add_argument("--output", help="Output path to test")
    parser.add_argument("--config", help="Pipeline configuration file")
    
    args = parser.parse_args()
    
    diagnostics = PipelineDiagnostics(args.config)
    results = diagnostics.run_full_diagnostic(args.input_path, args.output)
    
    # Save detailed results
    import json
    with open('diagnostic_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Detailed results saved to: diagnostic_results.json")


if __name__ == "__main__":
    main()
