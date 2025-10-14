import hashlib
import time
import httpx
from typing import List, Set
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client import models
from tqdm import tqdm

from eve.base_step import PipelineStep
from eve.model.document import Document


class VLLMEmbedder:
    """Client for VLLM embedding server."""

    def __init__(
        self, url: str, model_name: str, timeout: int = 300, api_key: str = "EMPTY"
    ):
        """Initialize VLLM embedder client.

        Args:
            url: Base URL of the VLLM server
            model_name: Name of the embedding model
            timeout: Request timeout in seconds
            api_key: API key for authentication (default: "EMPTY" for local servers)
        """
        self.url = url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.api_key = api_key

        # Set up headers with API key
        headers = (
            {"Authorization": f"Bearer {api_key}"}
            if api_key and api_key != "EMPTY"
            else {}
        )
        self.client = httpx.Client(timeout=timeout, headers=headers)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors

        Raises:
            Exception: If the API request fails
        """
        endpoint = f"{self.url}/v1/embeddings"

        payload = {"input": texts, "model": self.model_name, "encoding_format": "float"}

        try:
            response = self.client.post(endpoint, json=payload)
            response.raise_for_status()

            result = response.json()

            # Extract embeddings in order
            embeddings = [
                item["embedding"]
                for item in sorted(result["data"], key=lambda x: x["index"])
            ]

            return embeddings

        except httpx.HTTPError as e:
            raise Exception(f"VLLM embedding request failed: {e}")
        except KeyError as e:
            raise Exception(f"Unexpected response format from VLLM server: {e}")

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()


class QdrantUploadStep(PipelineStep):
    """Pipeline step for uploading chunked documents to Qdrant vector database."""

    def __init__(self, config: dict, name: str = "QdrantUpload"):
        """Initialize the Qdrant upload step.

        Args:
            config: Configuration containing:
                - vector_store.url: Qdrant instance URL
                - vector_store.api_key: API key for Qdrant authentication (optional)
                - collection_name: Target collection name
                - embedder.url: URL of VLLM embedding server
                - embedder.model_name: Embedding model identifier
                - embedder.timeout: Optional request timeout (default: 300)
                - embedder.api_key: Optional API key for VLLM authentication (default: "EMPTY")
                - batch_size: Number of documents per batch
                - vector_size: Dimension of embedding vectors
            name: Name for logging purposes
        """
        super().__init__(config, name)

        # Get vector store configuration
        vector_store_cfg = config.get("vector_store", {})
        self.qdrant_url = vector_store_cfg.get("url", "http://localhost:6333")
        self.vector_store_api_key = vector_store_cfg.get("api_key")

        # Get collection configuration
        self.collection_name = vector_store_cfg["collection_name"]
        self.batch_size = vector_store_cfg["batch_size"]
        self.vector_size = vector_store_cfg["vector_size"]

        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url, api_key=self.vector_store_api_key
        )

        # Initialize VLLM embedder
        embedding_cfg = config["embedder"]
        self.embedder = VLLMEmbedder(
            url=embedding_cfg["url"],
            model_name=embedding_cfg["model_name"],
            timeout=embedding_cfg.get("timeout", 300),
            api_key=embedding_cfg.get("api_key", "EMPTY"),
        )

        self.logger.info(f"Initialized VLLM embedder at {embedding_cfg['url']}")

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it doesn't exist."""
        if self.client.collection_exists(self.collection_name):
            self.logger.info(f"Collection '{self.collection_name}' already exists")
            return

        self.logger.info(f"Creating collection '{self.collection_name}'")

        # Create collection with optimized settings
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            shard_number=8,
            on_disk_payload=True,
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(always_ram=False)
            ),
        )

        # Update HNSW configuration
        self.client.update_collection(
            collection_name=self.collection_name,
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=128,
                full_scan_threshold=10_000,
                max_indexing_threads=2,
                on_disk=True,
            ),
        )

        # Update optimizer configuration
        self.client.update_collection(
            collection_name=self.collection_name,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=5000,
                deleted_threshold=0.2,
                vacuum_min_vector_number=1000,
                default_segment_number=2,
                max_segment_size=5_000_000,
                max_optimization_threads=2,
            ),
        )

        # Create payload indexes
        self._create_payload_indexes()

        self.logger.info("Collection created and optimized")

    def _create_payload_indexes(self) -> None:
        """Create indexes on payload fields for efficient filtering."""
        # Text index for title
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="title",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=50,
                lowercase=True,
            ),
        )

        # Integer indexes
        for field in ["year", "n_citations"]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema="integer",
            )

        # Text index for journal
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="journal",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                min_token_len=1,
                max_token_len=50,
                lowercase=True,
            ),
        )

    @staticmethod
    def _string_to_uint(s: str) -> int:
        """Convert string to unsigned integer using SHA256 hash."""
        hash_bytes = hashlib.sha256(s.encode("utf-8")).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)

    def _get_existing_ids(self) -> Set[int]:
        """Retrieve all existing point IDs from the collection."""
        existing_ids = set()
        scroll_offset = None

        while True:
            response = self.client.scroll(
                collection_name=self.collection_name,
                offset=scroll_offset,
                limit=10000,
                with_payload=False,
                with_vectors=False,
            )
            for point in response[0]:
                existing_ids.add(point.id)

            if response[1] is None:
                break
            scroll_offset = response[1]

        return existing_ids

    def _upload_batch(
        self, batch_ids: List[int], batch_chunks: List[str], batch_metadata: List[dict]
    ) -> None:
        """Upload a batch of documents to Qdrant.

        Args:
            batch_ids: List of unique IDs
            batch_chunks: List of text chunks
            batch_metadata: List of metadata dictionaries
        """
        try:
            batch_vectors = self.embedder.embed_documents(batch_chunks)
        except Exception as e:
            self.logger.error(f"Embedding error: {e}")
            return

        points = [
            PointStruct(id=id_val, vector=vec, payload=meta)
            for id_val, vec, meta in zip(batch_ids, batch_vectors, batch_metadata)
        ]

        for attempt in range(3):
            try:
                self.client.upload_points(
                    collection_name=self.collection_name,
                    points=points,
                    parallel=10,
                    max_retries=3,
                )
                return
            except Exception as e:
                self.logger.error(f"Error uploading batch (attempt {attempt + 1}): {e}")
                time.sleep(10)
                if attempt < 2:
                    self.logger.info("Retrying...")
                else:
                    self.logger.warning("Skipping batch after 3 failed attempts")

    def _prepare_metadata(self, doc: Document) -> dict:
        """Prepare metadata from Document object for Qdrant storage.

        Args:
            doc: Document object

        Returns:
            Dictionary with cleaned metadata
        """
        meta = doc.metadata.copy()

        # Ensure year is an integer
        if "year" in meta:
            try:
                meta["year"] = int(float(meta["year"]))
            except (ValueError, TypeError):
                meta["year"] = None

        # Ensure title is a string
        if "title" in meta:
            meta["title"] = str(meta["title"])

        # Add content if not present
        if "content" not in meta:
            meta["content"] = doc.content

        # Add file metadata
        meta["file_path"] = str(doc.file_path)
        meta["file_format"] = doc.file_format
        meta["filename"] = doc.filename

        return meta

    async def execute(self, documents: List[Document]) -> List[Document]:
        """Execute the upload step.

        Args:
            documents: List of Document objects to upload

        Returns:
            The same list of documents (pass-through for chaining)
        """
        if not documents:
            self.logger.warning("No documents to upload")
            return documents

        self.logger.info(f"Processing {len(documents)} documents")

        # Prepare data for upload
        ids = []
        chunks = []
        metadata = []

        for doc in documents:
            # Create unique ID based on file path and content hash
            doc_id = (
                f"{doc.filename}_{hashlib.md5(doc.content.encode()).hexdigest()[:8]}"
            )
            ids.append(doc_id)
            chunks.append(doc.content)
            metadata.append(self._prepare_metadata(doc))

        # Convert to uint IDs
        uint_ids = [self._string_to_uint(id_str) for id_str in ids]
        to_process = list(zip(uint_ids, chunks, metadata, ids))

        # Filter out existing IDs
        existing_ids = self._get_existing_ids()
        to_process = [item for item in to_process if item[0] not in existing_ids]

        skipped = len(uint_ids) - len(to_process)
        self.logger.info(f"Skipping {skipped} existing documents")
        self.logger.info(f"Uploading {len(to_process)} new vectors")

        # Upload in batches
        for i in tqdm(
            range(0, len(to_process), self.batch_size),
            desc=f"Uploading to {self.collection_name}",
        ):
            batch = to_process[i : i + self.batch_size]
            batch_ids = [item[0] for item in batch]
            batch_chunks = [item[1] for item in batch]
            batch_metadata = [item[2] for item in batch]

            self._upload_batch(batch_ids, batch_chunks, batch_metadata)

        self.logger.info(f"Successfully uploaded {len(to_process)} documents")

        # Return documents for potential further processing
        return documents
