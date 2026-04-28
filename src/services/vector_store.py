"""Vector store service for TeCoD application."""

import time
from typing import Any

import pandas as pd

from ..exceptions.base import ServiceInitializationError, VectorStoreError
from ..utils.timing import log_with_time_elapsed
from .base import Service
from .embedding import EmbeddingService


class VectorStoreService(Service):
    """Service for managing vector store operations using Milvus."""

    def __init__(self, config, embedding_service: EmbeddingService, logger=None):
        super().__init__(config, logger)
        self.embedding_service = embedding_service
        self._client = None

    def initialize(self) -> None:
        """Initialize the Milvus client."""
        try:
            with log_with_time_elapsed("Vector store initialization", self.logger):
                from pymilvus import MilvusClient

                index_path = self.config.index_path
                self.logger.info(f"Initializing vector store at: {index_path}")

                if not index_path.exists():
                    raise VectorStoreError("initialize", f"Index not found at {index_path}")

                self._client = MilvusClient(str(index_path))
                self._mark_initialized()

                self.logger.info("Vector store service initialized")

        except Exception as e:
            self.logger.exception(f"Failed to initialize VectorStoreService: {str(e)}")
            raise ServiceInitializationError("VectorStoreService", str(e)) from e

    def cleanup(self) -> None:
        """Cleanup vector store resources."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                self.logger.debug("Cleanup error closing Milvus client", exc_info=True)
            self._client = None
        self._initialized = False

    def create_index(self, examples_data: pd.DataFrame) -> None:
        """Create a new vector index from examples data.

        Args:
            examples_data: DataFrame containing examples with self.config.emb.masked_nlq_key column

        Raises:
            VectorStoreError: If index creation fails
        """
        try:
            from pymilvus import DataType, MilvusClient

            self.logger.info("Creating new vector index")

            self.embedding_service.ensure_initialized()

            # Validate the input DataFrame up-front so callers get a
            # contextful error instead of a bare KeyError.
            masked_key = self.config.emb.masked_nlq_key
            if masked_key not in examples_data.columns:
                raise VectorStoreError(
                    "create_index",
                    f"examples_data is missing configured masked_nlq_key "
                    f"{masked_key!r}. Columns present: {list(examples_data.columns)}",
                )
            if len(examples_data) == 0:
                raise VectorStoreError(
                    "create_index",
                    "examples_data is empty; cannot build a vector index.",
                )

            # Generate embeddings for the examples
            embeddings = self.embedding_service.embed(
                examples_data[masked_key].tolist(), prompt="skeleton"
            )
            emb_dim = embeddings.shape[-1]

            # Create new client (will create the index file)
            index_path = self.config.index_path
            index_path.parent.mkdir(parents=True, exist_ok=True)

            if index_path.exists():
                # Remove existing index
                index_path.unlink()

            client = MilvusClient(str(index_path))
            try:
                collection_name = self.config.emb.collection_name

                # Drop existing collection if it exists
                if client.has_collection(collection_name):
                    client.drop_collection(collection_name)

                # Create schema
                schema = client.create_schema(enable_dynamic_field=True)
                schema.add_field("id", DataType.INT64, is_primary=True, auto_id=False)
                schema.add_field(
                    self.config.emb.emb_field_name,
                    DataType.FLOAT_VECTOR,
                    dim=emb_dim,
                )

                # Create index parameters
                index_params = client.prepare_index_params()
                index_params.add_index(
                    primary_field_name="id",
                    field_name=self.config.emb.emb_field_name,
                    index_name=self.config.emb.index_name,
                    index_type="FLAT",
                    metric_type="COSINE",
                )

                # Create collection
                client.create_collection(
                    schema=schema, collection_name=collection_name, index_params=index_params
                )

                # Prepare data for insertion
                data_to_insert = []
                for i, emb in enumerate(embeddings):
                    data_to_insert.append(
                        {
                            "id": int(i),
                            self.config.emb.emb_field_name: emb.tolist(),
                        }
                    )

                # Insert data
                client.insert(collection_name, data_to_insert)
                client.flush(collection_name)

                # Update the client reference
                if self._client is not None:
                    self._client.close()
                self._client = client

                if not self._initialized:
                    self._mark_initialized()

                self.logger.info(f"Vector index created with {len(data_to_insert)} entries")
            except Exception:
                client.close()
                raise

        except Exception as e:
            self.logger.exception(f"Failed to create vector index: {str(e)}")
            raise VectorStoreError("create_index", str(e)) from e

    def search(
        self, query: str, top_k: int = None, *, timing_data: dict | None = None
    ) -> list[pd.DataFrame]:
        """Search for similar vectors.

        Args:
            query: Query text to search for
            top_k: Number of top results to return (defaults to config.tecod.vectorsearch_top_k)
            timing_data: When provided, timing entries are recorded into this dict

        Returns:
            List of DataFrames containing search results

        Raises:
            VectorStoreError: If search fails
        """
        self.ensure_initialized()

        if top_k is None:
            top_k = self.config.tecod.vectorsearch_top_k

        try:
            self.embedding_service.ensure_initialized()

            # Generate query embedding
            embedding_start = time.perf_counter()
            query_embedding = self.embedding_service.embed(query, prompt="query")
            embedding_elapsed = time.perf_counter() - embedding_start
            if timing_data is not None:
                timing_data["query_embedding"] = embedding_elapsed
            else:
                self.logger.debug(
                    f"Query embedding generation... done in {embedding_elapsed * 1000:.1f}ms"
                )

            # Perform search
            search_start = time.perf_counter()
            results = self._client.search(
                collection_name=self.config.emb.collection_name,
                data=query_embedding,
                limit=top_k,
                anns_field=self.config.emb.emb_field_name,
                output_fields=["idx"],
            )
            search_elapsed = time.perf_counter() - search_start
            if timing_data is not None:
                timing_data["vector_search_query"] = search_elapsed
            else:
                self.logger.debug(
                    f"Vector search (top_k={top_k})... done in {search_elapsed * 1000:.1f}ms"
                )

            # Convert results to DataFrames
            processing_start = time.perf_counter()
            output = [pd.DataFrame(x).drop(columns=["entity"]).set_index("id") for x in results]
            processing_elapsed = time.perf_counter() - processing_start
            if timing_data is not None:
                timing_data["search_result_processing"] = processing_elapsed
            else:
                self.logger.debug(
                    f"Search results processing... done in {processing_elapsed * 1000:.1f}ms"
                )

            return output

        except Exception as e:
            self.logger.exception(f"Failed to search vector store: {str(e)}")
            raise VectorStoreError("search", str(e)) from e

    def has_collection(self, collection_name: str | None = None) -> bool:
        """Check if collection exists.

        Args:
            collection_name: Collection name to check (defaults to config collection)

        Returns:
            True if collection exists
        """
        self.ensure_initialized()

        collection_name = collection_name or self.config.emb.collection_name
        return self._client.has_collection(collection_name)

    def get_collection_info(self) -> dict[str, Any]:
        """Get information about the current collection.

        Returns:
            Dictionary containing collection information
        """
        self.ensure_initialized()

        try:
            collection_name = self.config.emb.collection_name
            if not self.has_collection(collection_name):
                raise VectorStoreError(
                    "get_collection_info", f"Collection {collection_name} does not exist"
                )

            # This is a simplified version - Milvus client methods may vary
            return {
                "collection_name": collection_name,
                "exists": True,
                "embedding_field": self.config.emb.emb_field_name,
                "index_name": self.config.emb.index_name,
            }

        except Exception as e:
            self.logger.exception(f"Failed to get collection info: {str(e)}")
            raise VectorStoreError("get_collection_info", str(e)) from e

    @property
    def client(self):
        """Get the underlying Milvus client."""
        self.ensure_initialized()
        return self._client
