"""
embedder_server.py

Simple wrapper to start VLLM embedding server with YAML configuration.
"""

import argparse
import sys
import subprocess
import signal
import os
from pathlib import Path


class VLLMEmbeddingServer:
    """Manages a VLLM embedding server instance."""

    def __init__(self, config_path: str):
        """Initialize the embedding server.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        self.process = None

    def start(self):
        """Start the VLLM embedding server."""
        cmd = ["vllm", "serve", "--config", str(self.config_path)]

        print("=" * 80)
        print("Starting VLLM Embedding Server")
        print("=" * 80)
        print(f"Configuration: {self.config_path}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 80)
        print()

        try:
            # Start the server process
            self.process = subprocess.Popen(
                cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                preexec_fn=os.setsid if sys.platform != "win32" else None,
            )

            # Wait for the process
            self.process.wait()

        except KeyboardInterrupt:
            print("\nReceived interrupt signal. Shutting down server...")
            self.stop()
        except Exception as e:
            print(f"Error starting server: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop the VLLM server gracefully."""
        if self.process:
            print("Stopping VLLM server...")
            try:
                if sys.platform != "win32":
                    # Send SIGTERM to the process group
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                else:
                    self.process.terminate()

                # Wait for graceful shutdown
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Force killing server...")
                if sys.platform != "win32":
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
            except Exception as e:
                print(f"Error stopping server: {e}")
            finally:
                self.process = None


def main():
    parser = argparse.ArgumentParser(
        description="Start a VLLM embedding server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python embedder_server.py --config configs/qwen_embedder.yaml

The server will expose an OpenAI-compatible API at the configured host:port
  Default: http://localhost:8000/v1/embeddings
        """,
    )

    parser.add_argument(
        "--config", type=str, required=True, help="Path to VLLM YAML configuration file"
    )

    args = parser.parse_args()

    # Create and start server
    server = VLLMEmbeddingServer(args.config)

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutdown complete.")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
