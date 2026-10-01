import json
import logging
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from vector_brain.common.filters import make_indexes
from vector_brain.common.func_tools import make_partial_function
from vector_brain.common.wrap_filters import wrap_filters
from vector_brain.embeddings.base import EmbeddingProvider
from vector_brain.embeddings.factory import create_embedding_provider
from vector_brain.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from vector_brain.settings import (
    DEFAULT_TOOL_DELETE_DESCRIPTION,
    DEFAULT_TOOL_EDIT_DESCRIPTION,
    DEFAULT_TOOL_FIND_DESCRIPTION,
    DEFAULT_TOOL_STORE_DESCRIPTION,
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
    generate_delete_description,
    generate_edit_description,
    generate_find_description,
    generate_store_description,
)

logger = logging.getLogger(__name__)


class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant with per-collection metadata schemas.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "vector-brain",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        # Parse metadata schemas for multi-collection support
        self._schemas = qdrant_settings.metadata_schemas
        self._has_schemas = len(self._schemas) > 0

        # Auto-generate tool descriptions from schemas if METADATA_SCHEMAS is set
        # and the user hasn't explicitly overridden via TOOL_*_DESCRIPTION env vars
        if self._has_schemas:
            if (
                self.tool_settings.tool_store_description
                == DEFAULT_TOOL_STORE_DESCRIPTION
            ):
                self.tool_settings.tool_store_description = generate_store_description(
                    self._schemas
                )
            if (
                self.tool_settings.tool_find_description
                == DEFAULT_TOOL_FIND_DESCRIPTION
            ):
                self.tool_settings.tool_find_description = generate_find_description(
                    self._schemas
                )
            if (
                self.tool_settings.tool_delete_description
                == DEFAULT_TOOL_DELETE_DESCRIPTION
            ):
                self.tool_settings.tool_delete_description = (
                    generate_delete_description(self._schemas)
                )
            if (
                self.tool_settings.tool_edit_description
                == DEFAULT_TOOL_EDIT_DESCRIPTION
            ):
                self.tool_settings.tool_edit_description = generate_edit_description(
                    self._schemas
                )

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
            metadata_schemas=self._schemas,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def _write_error(self, collection_name: str) -> str | None:
        """
        Return an error message if writes to the collection are not allowed, otherwise None.

        In multi-collection mode only collections defined (and enabled) in the schemas
        accept writes, so a mistyped collection name cannot silently create a new one.
        """
        if not self._has_schemas:
            if self.qdrant_settings.read_only:
                return f"Error: collection '{collection_name}' is read-only."
            return None
        schema = self._schemas.get(collection_name)
        if schema is None:
            writable = (
                ", ".join(self.qdrant_settings.writable_collection_names) or "none"
            )
            return (
                f"Error: unknown collection '{collection_name}'. "
                f"Writable collections: {writable}."
            )
        if schema.read_only:
            return f"Error: collection '{collection_name}' is read-only."
        return None

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        # Build collection name descriptions for the LLM
        if self._has_schemas:
            all_names = ", ".join(self.qdrant_settings.all_collection_names)
            writable_names = ", ".join(self.qdrant_settings.writable_collection_names)
            find_collection_desc = f"Collection to search in ({all_names})"
            store_collection_desc = f"Collection to store in ({writable_names})"
            delete_collection_desc = f"Collection to delete from ({writable_names})"
            edit_collection_desc = f"Collection to edit in ({writable_names})"
        else:
            find_collection_desc = "The collection to search in"
            store_collection_desc = "The collection to store the information in"
            delete_collection_desc = "The collection to delete from"
            edit_collection_desc = "The collection to edit the information in"

        write_error = self._write_error

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[str, Field(description=store_collection_desc)],
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """Store some information in Qdrant."""
            # Only configured, writable collections accept writes
            if error := write_error(collection_name):
                return error

            await ctx.debug(f"Storing information {information} in Qdrant")

            # Always ensure created_at timestamp exists in metadata.
            # Copy so that stamping created_at does not mutate the caller's metadata
            metadata = dict(metadata or {})
            if "created_at" not in metadata:
                metadata["created_at"] = datetime.now(timezone.utc).isoformat()

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            return f"Remembered: {information} in collection {collection_name}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[str, Field(description=find_collection_desc)],
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """Find memories in Qdrant."""
            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def delete(
            ctx: Context,
            query: Annotated[
                str,
                Field(description="Semantic search query to find the memory to delete"),
            ],
            collection_name: Annotated[str, Field(description=delete_collection_desc)],
        ) -> str:
            """Delete a memory from Qdrant by semantic search."""
            # Only configured, writable collections accept writes
            if error := write_error(collection_name):
                return error

            await ctx.debug(f"Deleting memory matching '{query}'")
            deleted = await self.qdrant_connector.delete(
                query, collection_name=collection_name
            )
            if not deleted:
                return f"No matching memory found for: {query}"
            contents = [entry.content[:100] for entry in deleted]
            return f"Deleted {len(deleted)} memory(ies): {contents}"

        async def edit(
            ctx: Context,
            query: Annotated[
                str,
                Field(
                    description="Semantic search query used to find the existing memory to update"
                ),
            ],
            information: Annotated[str, Field(description="Replacement text to store")],
            collection_name: Annotated[str, Field(description=edit_collection_desc)],
            metadata: Annotated[
                Metadata | None,
                Field(
                    description=(
                        "Replacement metadata for the memory. If omitted, the existing metadata is preserved."
                    )
                ),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
        ) -> str:
            """Edit a memory in Qdrant by replacing the closest semantic match."""
            # Only configured, writable collections accept writes
            if error := write_error(collection_name):
                return error

            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Editing memory matching '{query}'")

            updated_entry = await self.qdrant_connector.edit(
                query,
                Entry(content=information, metadata=metadata),
                collection_name=collection_name,
                query_filter=query_filter,
            )
            if not updated_entry:
                return f"No matching memory found for: {query}"
            return f"Updated memory matching '{query}' in collection {collection_name}"

        find_foo = find
        store_foo = store
        delete_foo = delete
        edit_foo = edit

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
            edit_foo = wrap_filters(edit_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})
            edit_foo = make_partial_function(edit_foo, {"query_filter": None})

        # If a single default collection is set (legacy mode), fix collection_name
        # But NOT if METADATA_SCHEMAS is set (multi-collection mode)
        if self.qdrant_settings.collection_name and not self._has_schemas:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            delete_foo = make_partial_function(
                delete_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            edit_foo = make_partial_function(
                edit_foo, {"collection_name": self.qdrant_settings.collection_name}
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        if not self.qdrant_settings.read_only:
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
            self.tool(
                delete_foo,
                name="qdrant-delete",
                description=self.tool_settings.tool_delete_description,
            )
            self.tool(
                edit_foo,
                name="qdrant-edit",
                description=self.tool_settings.tool_edit_description,
            )
