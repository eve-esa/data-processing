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
    """Client for VLLM embedding server.

    This class provides a simple interface to interact with a VLLM server
    that exposes an OpenAI-compatible embeddings API endpoint.
    """

    def __init__(
        self, url: str, model_name: str, timeout: int = 300, api_key: str = "EMPTY"
    ):
        """Initialize VLLM embedder client.

        Args:
            url (str): Base URL of the VLLM server.
            model_name (str): Name of the embedding model to use.
            timeout (int, optional): Request timeout in seconds. Defaults to 300.
            api_key (str, optional): API key for authentication. Use "EMPTY" for
                local servers. Defaults to "EMPTY".
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

    def embed_documents(self, texts: List[str]) -> List[List[float] | None]:
        """Generate embeddings for a list of texts.

        Sends requests to the VLLM server's /v1/embeddings endpoint to generate
        embeddings for each text. Failed requests return None for that text.

        Args:
            texts (List[str]): List of text strings to embed.

        Returns:
            List[List[float] | None]: List of embedding vectors. Each element is
                either a list of floats (the embedding) or None if embedding failed.
        """
        embeddings = []
        for text in texts:
            endpoint = f"{self.url}/v1/embeddings"

            payload = {"input": [text], "model": self.model_name, "encoding_format": "float"}

            try:
                response = self.client.post(endpoint, json=payload)
                response.raise_for_status()

                result = response.json()

                # Extract embedding (single document)
                embedding = result["data"][0]["embedding"]
                embeddings.append(embedding)

            except httpx.HTTPError as e:
                print(f"VLLM embedding request failed: {e}")
                embeddings.append(None)
            except KeyError as e:
                embeddings.append(None)
                print(f"Unexpected response format from VLLM server: {e}")
        return embeddings

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()


class QdrantUploadStep(PipelineStep):
    """Pipeline step for uploading chunked documents to Qdrant vector database or storing embeddings locally.

    Supports two modes:
    
    - "qdrant": Upload embeddings to a Qdrant vector database
    - "local": Store embeddings in document metadata without uploading
    """

    def __init__(self, config: dict, name: str = "QdrantUpload"):
        """Initialize the Qdrant upload step.

        Args:
            config (dict): Configuration dictionary containing:
                - mode (str, optional): "qdrant" or "local". Defaults to "qdrant".
                - use_existing_embeddings (bool, optional): If True, use embeddings
                    from document.embedding field. Defaults to False.
                - upload_pipeline_metadata (bool, optional): If True, include
                    pipeline_metadata in Qdrant payload. Defaults to False.
                - vector_store (dict, required for "qdrant" mode):
                    - url (str): Qdrant instance URL.
                    - api_key (str, optional): API key for Qdrant authentication.
                    - collection_name (str): Target collection name.
                    - batch_size (int): Number of documents per batch.
                    - vector_size (int): Dimension of embedding vectors.
                - embedder (dict, required if use_existing_embeddings=False):
                    - url (str): URL of VLLM embedding server.
                    - model_name (str): Embedding model identifier.
                    - timeout (int, optional): Request timeout in seconds. Defaults to 300.
                    - api_key (str, optional): API key for VLLM. Defaults to "EMPTY".
                - batch_size (int, optional): Batch size for local mode. Defaults to 10.
            name (str, optional): Name for logging purposes. Defaults to "QdrantUpload".

        Raises:
            ValueError: If mode is not "qdrant" or "local".
        """
        super().__init__(config, name)

        # Determine mode: "qdrant" or "local"
        self.mode = config.get("mode", "qdrant").lower()

        if self.mode not in ["qdrant", "local"]:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'qdrant' or 'local'")

        # Check if we should use existing embeddings from document.embedding field
        self.use_existing_embeddings = config.get("use_existing_embeddings", False)

        # Check if we should upload pipeline_metadata to Qdrant
        self.upload_pipeline_metadata = config.get("upload_pipeline_metadata", False)

        # Initialize VLLM embedder only if not using existing embeddings
        if not self.use_existing_embeddings:
            embedding_cfg = config["embedder"]
            self.embedder = VLLMEmbedder(
                url=embedding_cfg["url"],
                model_name=embedding_cfg["model_name"],
                timeout=embedding_cfg.get("timeout", 300),
                api_key=embedding_cfg.get("api_key", "EMPTY"),
            )
            self.logger.info(f"Initialized VLLM embedder at {embedding_cfg['url']}")
        else:
            self.embedder = None
            self.logger.info("Using existing embeddings from document metadata")

        self.logger.info(f"Mode: {self.mode}")

        # Initialize Qdrant-specific configuration only if mode is "qdrant"
        if self.mode == "qdrant":
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

            # Ensure collection exists
            self._ensure_collection()
            self.existing_ids = self._get_existing_ids()
        else:
            # Local mode: set batch size for processing
            self.batch_size = config.get("batch_size", 10)
            self.client = None

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it doesn't exist.

        Creates a new collection with optimized settings including HNSW indexing,
        binary quantization, and on-disk storage. Also creates payload indexes
        for efficient filtering.
        """
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
        """Create indexes on payload fields for efficient filtering.

        Creates text indexes for 'title' and 'journal' fields, and integer
        indexes for 'year' and 'n_citations' fields to enable fast filtering
        and searching on these metadata fields.
        """
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
        """Convert string to unsigned integer using SHA256 hash.

        Args:
            s (str): Input string to hash.

        Returns:
            int: Unsigned 64-bit integer derived from the hash.
        """
        hash_bytes = hashlib.sha256(s.encode("utf-8")).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big", signed=False)

    def _get_existing_ids(self) -> Set[int]:
        """Retrieve all existing point IDs from the collection with retry logic.

        Scrolls through the entire collection to fetch all point IDs. Implements
        retry logic with exponential backoff to handle temporary failures.

        Returns:
            Set[int]: Set of existing point IDs in the collection.

        Raises:
            Exception: If scroll request fails after max retries.
        """
        existing_ids = set()
        scroll_offset = None
        max_retries = 3
        retry_delay = 5

        while True:
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.client.scroll(
                        collection_name=self.collection_name,
                        offset=scroll_offset,
                        limit=10000,
                        with_payload=False,
                        with_vectors=False,
                        timeout=3000,
                    )
                    break  # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Scroll request failed (attempt {attempt + 1}/{max_retries}): {e}")
                        self.logger.info(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        self.logger.error(f"Scroll request failed after {max_retries} attempts: {e}")
                        raise

            for point in response[0]:
                existing_ids.add(point.id)

            if response[1] is None:
                break
            scroll_offset = response[1]

        return existing_ids

    def _upload_batch(
        self, batch_ids: List[int], batch_chunks: List[str], batch_metadata: List[dict], batch_embeddings: List[List[float]] = None
    ) -> None:
        """Upload a batch of documents to Qdrant.

        Generates embeddings (if not provided) and uploads points to Qdrant.
        Implements retry logic with up to 3 attempts on failure.

        Args:
            batch_ids (List[int]): List of unique point IDs.
            batch_chunks (List[str]): List of text chunks to embed.
            batch_metadata (List[dict]): List of metadata dictionaries for each point.
            batch_embeddings (List[List[float]], optional): Pre-computed embeddings
                to use instead of generating new ones. Defaults to None.
        """
        # Generate embeddings if not provided
        if batch_embeddings is None:
            if self.use_existing_embeddings:
                self.logger.error("use_existing_embeddings=True but no embeddings provided")
                return

            try:
                batch_vectors = self.embedder.embed_documents(batch_chunks)
            except Exception as e:
                self.logger.error(f"Embedding error: {e}")
                return
        else:
            batch_vectors = batch_embeddings

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

        Extracts and cleans metadata fields, performing type conversions and
        formatting for Qdrant compatibility. Includes document content, user
        metadata (unwrapped), and optionally pipeline metadata (wrapped).

        Args:
            doc (Document): Document object containing content and metadata.

        Returns:
            dict: Dictionary with cleaned metadata ready for Qdrant payload.
                Includes 'content' field, all metadata fields at root level,
                and optionally 'pipeline_metadata' as a nested dict.
        """
        payload = {}

        # Add content
        payload["content"] = doc.content

        # Add original metadata fields directly to root level (unwrapped)
        if doc.metadata:
            # Clean up metadata types
            metadata_copy = doc.metadata.copy()

            # Ensure year is an integer if present
            if "year" in metadata_copy:
                try:
                    metadata_copy["year"] = int(float(metadata_copy["year"]))
                except (ValueError, TypeError):
                    metadata_copy["year"] = None

            # Ensure title is a string if present
            if "title" in metadata_copy:
                metadata_copy["title"] = str(metadata_copy["title"])

            # Add all metadata fields directly to payload (unwrapped)
            payload.update(metadata_copy)

        # Add pipeline_metadata as wrapped dict if configured
        if self.upload_pipeline_metadata and doc.pipeline_metadata:
            payload["pipeline_metadata"] = doc.pipeline_metadata.copy()

        return payload

    async def execute(self, documents: List[Document]) -> List[Document]:
        """Execute the upload step.

        Routes to appropriate execution method based on configured mode
        (local or qdrant).

        Args:
            documents (List[Document]): List of Document objects to process.

        Returns:
            List[Document]: The same list of documents passed through for
                pipeline chaining.
        """
        if not documents:
            self.logger.warning("No documents to process")
            return documents

        self.logger.info(f"Processing {len(documents)} documents")

        if self.mode == "local":
            # Local mode: add embeddings to document metadata
            return await self._execute_local(documents)
        else:
            # Qdrant mode: upload to vector database
            return await self._execute_qdrant(documents)

    async def _execute_local(self, documents: List[Document]) -> List[Document]:
        """Execute local embedding storage.

        Generates embeddings for documents and stores them in the document.embedding
        field without uploading to Qdrant. Processes documents in configurable batches.

        Args:
            documents (List[Document]): List of Document objects to process.

        Returns:
            List[Document]: Documents with embeddings added to embedding field.
        """
        self.logger.info("Generating embeddings in local mode")

        # Process in batches
        for i in tqdm(
            range(0, len(documents), self.batch_size),
            desc="Generating embeddings",
        ):
            batch = documents[i : i + self.batch_size]
            batch_texts = [doc.content for doc in batch]

            try:
                # Generate embeddings
                batch_embeddings = self.embedder.embed_documents(batch_texts)

                # Add embeddings to document.embedding field
                for doc, embedding in zip(batch, batch_embeddings):
                    doc.embedding = embedding

            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch: {e}")
                # Continue with next batch

        self.logger.info(f"Successfully generated embeddings for {len(documents)} documents")
        return documents

    async def _execute_qdrant(self, documents: List[Document]) -> List[Document]:
        """Execute Qdrant upload mode.

        Prepares document data, generates or extracts embeddings, filters out
        existing documents, and uploads to Qdrant in batches.

        Args:
            documents (List[Document]): List of Document objects to upload.

        Returns:
            List[Document]: The same list of documents passed through for
                pipeline chaining.
        """
        # Prepare data for upload
        ids = []
        chunks = []
        metadata = []
        embeddings = [] if self.use_existing_embeddings else None

        for doc in documents:
            # Create unique ID based on file path and content hash
            doc_id = (
                f"{doc.filename}_{hashlib.md5(doc.content.encode()).hexdigest()[:8]}"
            )
            ids.append(doc_id)
            chunks.append(doc.content)
            metadata.append(self._prepare_metadata(doc))

            # Extract existing embeddings if configured
            if self.use_existing_embeddings:
                if doc.embedding is None:
                    self.logger.error(f"Document {doc.filename} missing embedding")
                    # Skip this document
                    ids.pop()
                    chunks.pop()
                    metadata.pop()
                    continue
                embeddings.append(doc.embedding)

        # Convert to uint IDs
        uint_ids = [self._string_to_uint(id_str) for id_str in ids]

        if self.use_existing_embeddings:
            to_process = list(zip(uint_ids, chunks, metadata, embeddings, ids))
        else:
            to_process = list(zip(uint_ids, chunks, metadata, ids))

        # Filter out existing IDs
        if self.use_existing_embeddings:
            to_process = [item for item in to_process if item[0] not in self.existing_ids]
        else:
            to_process = [item for item in to_process if item[0] not in self.existing_ids]

        skipped = len(uint_ids) - len(to_process)
        self.logger.info(f"Skipping {skipped} existing documents")
        self.logger.info(f"Uploading {len(to_process)} new vectors")

        # Upload in batches
        for i in tqdm(
            range(0, len(to_process), self.batch_size),
            desc=f"Uploading to {self.collection_name}",
        ):
            batch = to_process[i : i + self.batch_size]

            if self.use_existing_embeddings:
                batch_ids = [item[0] for item in batch]
                batch_chunks = [item[1] for item in batch]
                batch_metadata = [item[2] for item in batch]
                batch_embeddings = [item[3] for item in batch]
                self._upload_batch(batch_ids, batch_chunks, batch_metadata, batch_embeddings)
            else:
                batch_ids = [item[0] for item in batch]
                batch_chunks = [item[1] for item in batch]
                batch_metadata = [item[2] for item in batch]
                self._upload_batch(batch_ids, batch_chunks, batch_metadata)

        self.logger.info(f"Successfully uploaded {len(to_process)} documents")

        # Return documents for potential further processing
        return documents
