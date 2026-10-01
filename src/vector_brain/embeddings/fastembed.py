import asyncio
import logging
import os
import shutil
import tempfile

from fastembed import TextEmbedding
from fastembed.common.model_description import DenseModelDescription

from vector_brain.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class FastEmbedProvider(EmbeddingProvider):
    """
    FastEmbed implementation of the embedding provider.
    :param model_name: The name of the FastEmbed model to use.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_model = self._load_model(model_name)

    def _load_model(self, model_name: str) -> TextEmbedding:
        """
        Load the FastEmbed model, tolerating a missing or corrupted local
        cache. FastEmbed caches downloaded models on disk; if that cache is
        partially written (e.g. the process was killed mid-download, or the
        snapshot files were deleted while the blobs remained) the underlying
        ONNX runtime raises a ``NoSuchFile`` error on load.

        Instead of crashing the whole MCP server, we detect that failure,
        purge the broken cache directory for this model, and let FastEmbed
        re-download it from scratch.
        """
        try:
            return TextEmbedding(model_name)
        except Exception as exc:  # noqa: BLE001 - we want to recover from anything
            logger.warning(
                "Failed to load embedding model %r from cache (%s). "
                "Purging cached files and re-downloading.",
                model_name,
                exc,
            )
            self._purge_model_cache(model_name)
            # Retry with a clean cache. If this fails too, let it propagate.
            return TextEmbedding(model_name)

    @staticmethod
    def _cache_dir() -> str:
        """
        Resolve the directory FastEmbed uses to cache models, mirroring
        FastEmbed's own resolution logic.
        """
        cache_path = os.environ.get("FASTEMBED_CACHE_PATH")
        if cache_path:
            return cache_path
        return os.path.join(tempfile.gettempdir(), "fastembed_cache")

    def _purge_model_cache(self, model_name: str) -> None:
        """
        Delete any cached files for ``model_name`` so the next load forces a
        fresh download. This handles both the HuggingFace-style layout
        (``models--org--name``) and the flat layout FastEmbed sometimes uses.
        """
        cache_dir = self._cache_dir()
        if not os.path.isdir(cache_dir):
            return

        # HuggingFace hub layout: "jinaai/jina-embeddings-v3" ->
        # "models--jinaai--jina-embeddings-v3"
        hf_dir_name = "models--" + model_name.replace("/", "--")

        # FastEmbed's flat layout uses the last path component.
        flat_dir_name = model_name.split("/")[-1]

        for candidate in {hf_dir_name, flat_dir_name}:
            target = os.path.join(cache_dir, candidate)
            if os.path.isdir(target):
                logger.info("Removing cached model directory: %s", target)
                shutil.rmtree(target, ignore_errors=True)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.passage_embed(documents))
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        return embeddings[0].tolist()

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Important: This is compatible with the FastEmbed logic used before 0.6.0.
        """
        model_name = self.embedding_model.model_name.split("/")[-1].lower()
        return f"fast-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        model_description: DenseModelDescription = (
            self.embedding_model._get_model_description(self.model_name)
        )
        return model_description.dim
