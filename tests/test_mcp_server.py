import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from fastmcp import Client
from fastmcp.client.transports import FastMCPTransport

from tests.conftest import (
    DISABLED_COLLECTIONS,
    ENABLED_COLLECTIONS,
    READ_ONLY_COLLECTIONS,
    WRITABLE_COLLECTIONS,
)
from vector_brain.embeddings.base import EmbeddingProvider
from vector_brain.mcp_server import QdrantMCPServer
from vector_brain.qdrant import Entry
from vector_brain.settings import (
    DEFAULT_TOOL_DELETE_DESCRIPTION,
    DEFAULT_TOOL_EDIT_DESCRIPTION,
    DEFAULT_TOOL_FIND_DESCRIPTION,
    DEFAULT_TOOL_STORE_DESCRIPTION,
    QdrantSettings,
    ToolSettings,
)

WRITE_TOOLS = {"qdrant-store", "qdrant-edit", "qdrant-delete"}
ALL_TOOLS = WRITE_TOOLS | {"qdrant-find"}


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[1.0] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        return [1.0]

    def get_vector_name(self) -> str:
        return "fake"

    def get_vector_size(self) -> int:
        return 1


@pytest.fixture
def make_server(monkeypatch):
    """Factory so tests can tweak env vars before the server is built."""

    def _make() -> QdrantMCPServer:
        monkeypatch.setenv("QDRANT_URL", ":memory:")
        return QdrantMCPServer(
            tool_settings=ToolSettings(),
            qdrant_settings=QdrantSettings(),
            embedding_provider=FakeEmbeddingProvider(),
        )

    return _make


@pytest.fixture
def server(make_server) -> QdrantMCPServer:
    return make_server()


async def call(server: QdrantMCPServer, tool: str, arguments: dict) -> str:
    """Call a tool through a real MCP client and return its text output."""
    async with Client(FastMCPTransport(server)) as client:
        result = await client.call_tool(tool, arguments)
    return "\n".join(getattr(content, "text", "") for content in result)


def collection_param(tool) -> dict:
    return tool.parameters["properties"]["collection_name"]


async def descriptions(server: QdrantMCPServer) -> dict[str, str]:
    tools = await server.get_tools()
    return {name: tool.description or "" for name, tool in tools.items()}


async def contents_in(server: QdrantMCPServer, collection: str) -> list[str]:
    entries = await server.qdrant_connector.search(
        "anything", collection_name=collection
    )
    return [entry.content for entry in entries]


# ──────────────────────────────────────────────
# Multi-collection mode (default)
# ──────────────────────────────────────────────


class TestMultiCollectionToolRegistration:
    async def test_all_tools_are_exposed(self, server):
        tools = await server.get_tools()
        assert set(tools) == ALL_TOOLS

    async def test_every_tool_requires_collection_name(self, server):
        tools = await server.get_tools()
        for name in ALL_TOOLS:
            assert "collection_name" in tools[name].parameters["required"], name

    async def test_collection_name_env_is_ignored(self, monkeypatch, make_server):
        """COLLECTION_NAME must not pin tools to one collection when schemas are configured."""
        monkeypatch.setenv("COLLECTION_NAME", "memories")

        tools = await make_server().get_tools()

        for name in ALL_TOOLS:
            assert "collection_name" in tools[name].parameters["required"], name

    async def test_find_parameter_lists_all_enabled_collections(self, server):
        tools = await server.get_tools()
        description = collection_param(tools["qdrant-find"])["description"]

        for collection in ENABLED_COLLECTIONS:
            assert collection in description
        for collection in DISABLED_COLLECTIONS:
            assert collection not in description

    async def test_write_parameters_list_only_writable_collections(self, server):
        tools = await server.get_tools()

        for name in WRITE_TOOLS:
            description = collection_param(tools[name])["description"]
            for collection in WRITABLE_COLLECTIONS:
                assert collection in description, name
            for collection in READ_ONLY_COLLECTIONS + DISABLED_COLLECTIONS:
                assert collection not in description, name

    async def test_query_filter_hidden_unless_arbitrary_filter_allowed(
        self, monkeypatch, make_server
    ):
        tools = await make_server().get_tools()
        assert "query_filter" not in tools["qdrant-find"].parameters["properties"]

        monkeypatch.setenv("QDRANT_ALLOW_ARBITRARY_FILTER", "true")
        tools = await make_server().get_tools()
        assert "query_filter" in tools["qdrant-find"].parameters["properties"]
        assert "query_filter" in tools["qdrant-edit"].parameters["properties"]


class TestMultiCollectionToolDescriptions:
    async def test_store_description_lists_writable_collections_and_fields(
        self, server
    ):
        description = (await descriptions(server))["qdrant-store"]

        assert (
            "collection_name=ai_thoughts: category (credential|personal|technical|general), "
            "tags, importance (low|medium|high|critical), confidence"
        ) in description
        assert "collection_name=project_notes: project, priority" in description
        assert "api_documentation" not in description
        assert "archived" not in description

    async def test_find_description_lists_all_collections_with_field_paths(
        self, server
    ):
        description = (await descriptions(server))["qdrant-find"]

        assert (
            "collection_name=ai_thoughts: filter by metadata.category, metadata.tags, "
            "metadata.importance, metadata.confidence"
        ) in description
        assert (
            "collection_name=project_notes: filter by metadata.project, metadata.priority"
        ) in description
        # Read-only collection is flagged and uses top-level field paths (field_prefix="")
        assert (
            "collection_name=api_documentation (search only): filter by type, method, path"
        ) in description
        assert "archived" not in description

    async def test_delete_and_edit_descriptions_list_only_writable_collections(
        self, server
    ):
        tool_descriptions = await descriptions(server)

        assert tool_descriptions["qdrant-delete"].endswith(
            "collection_name must be: ai_thoughts, project_notes"
        )
        assert tool_descriptions["qdrant-edit"].endswith(
            "collection_name must be: ai_thoughts, project_notes"
        )

    async def test_explicit_description_overrides_generated_one(
        self, monkeypatch, make_server
    ):
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")

        tools = await make_server().get_tools()

        assert tools["qdrant-find"].description == "Custom find description"
        # Others are still generated from the schemas
        assert tools["qdrant-store"].description != DEFAULT_TOOL_STORE_DESCRIPTION


class TestMultiCollectionRouting:
    async def test_store_and_find_in_writable_collection(self, server):
        result = await call(
            server,
            "qdrant-store",
            {
                "information": "User prefers dark mode",
                "collection_name": "ai_thoughts",
                "metadata": {"category": "personal"},
            },
        )
        assert result == "Remembered: User prefers dark mode in collection ai_thoughts"

        found = await call(
            server,
            "qdrant-find",
            {"query": "dark mode", "collection_name": "ai_thoughts"},
        )
        assert "User prefers dark mode" in found
        assert "personal" in found

    async def test_collections_are_isolated(self, server):
        await call(
            server,
            "qdrant-store",
            {"information": "A personal thought", "collection_name": "ai_thoughts"},
        )
        await call(
            server,
            "qdrant-store",
            {"information": "Ship v2 by Friday", "collection_name": "project_notes"},
        )

        assert await contents_in(server, "ai_thoughts") == ["A personal thought"]
        assert await contents_in(server, "project_notes") == ["Ship v2 by Friday"]

        found = await call(
            server,
            "qdrant-find",
            {"query": "anything", "collection_name": "project_notes"},
        )
        assert "Ship v2 by Friday" in found
        assert "A personal thought" not in found

    async def test_store_adds_created_at(self, server):
        await call(
            server,
            "qdrant-store",
            {"information": "Timestamped memory", "collection_name": "ai_thoughts"},
        )

        entries = await server.qdrant_connector.search(
            "anything", collection_name="ai_thoughts"
        )
        assert "created_at" in entries[0].metadata

    async def test_store_keeps_existing_created_at(self, server):
        await call(
            server,
            "qdrant-store",
            {
                "information": "Imported memory",
                "collection_name": "ai_thoughts",
                "metadata": {"created_at": "2020-01-01T00:00:00+00:00"},
            },
        )

        entries = await server.qdrant_connector.search(
            "anything", collection_name="ai_thoughts"
        )
        assert entries[0].metadata["created_at"] == "2020-01-01T00:00:00+00:00"

    async def test_store_does_not_modify_callers_metadata(self, server):
        """Reusing one metadata dict must give each memory its own created_at."""
        store_fn = (await server.get_tools())["qdrant-store"].fn
        ctx = AsyncMock()
        shared_metadata = {"category": "work"}

        await store_fn(ctx, "First", "ai_thoughts", shared_metadata)
        await asyncio.sleep(0.01)
        await store_fn(ctx, "Second", "ai_thoughts", shared_metadata)

        assert shared_metadata == {"category": "work"}
        entries = await server.qdrant_connector.search(
            "anything", collection_name="ai_thoughts"
        )
        created = {e.content: e.metadata["created_at"] for e in entries}
        assert created["First"] != created["Second"]

    async def test_edit_in_writable_collection_stamps_updated_at(self, server):
        await call(
            server,
            "qdrant-store",
            {
                "information": "Meeting on Monday",
                "collection_name": "project_notes",
                "metadata": {"project": "apollo"},
            },
        )

        result = await call(
            server,
            "qdrant-edit",
            {
                "query": "meeting",
                "information": "Meeting moved to Tuesday",
                "collection_name": "project_notes",
            },
        )
        assert result == "Updated memory matching 'meeting' in collection project_notes"

        entries = await server.qdrant_connector.search(
            "anything", collection_name="project_notes"
        )
        assert [e.content for e in entries] == ["Meeting moved to Tuesday"]
        assert entries[0].metadata["project"] == "apollo"
        assert "updated_at" in entries[0].metadata

    async def test_delete_in_writable_collection_only_affects_that_collection(
        self, server
    ):
        await call(
            server,
            "qdrant-store",
            {"information": "Delete me", "collection_name": "ai_thoughts"},
        )
        await call(
            server,
            "qdrant-store",
            {"information": "Keep me", "collection_name": "project_notes"},
        )

        result = await call(
            server,
            "qdrant-delete",
            {"query": "delete", "collection_name": "ai_thoughts"},
        )
        assert result == "Deleted 1 memory(ies): ['Delete me']"

        assert await contents_in(server, "ai_thoughts") == []
        assert await contents_in(server, "project_notes") == ["Keep me"]

    async def test_find_with_metadata_filter(self, monkeypatch, make_server):
        monkeypatch.setenv("QDRANT_ALLOW_ARBITRARY_FILTER", "true")
        server = make_server()
        for text, category in [
            ("API key is abc", "credential"),
            ("Likes tea", "personal"),
        ]:
            await call(
                server,
                "qdrant-store",
                {
                    "information": text,
                    "collection_name": "ai_thoughts",
                    "metadata": {"category": category},
                },
            )

        found = await call(
            server,
            "qdrant-find",
            {
                "query": "anything",
                "collection_name": "ai_thoughts",
                "query_filter": {
                    "must": [
                        {"key": "metadata.category", "match": {"value": "credential"}}
                    ]
                },
            },
        )

        assert "API key is abc" in found
        assert "Likes tea" not in found


class TestPerCollectionReadOnly:
    @pytest.fixture
    async def populated_server(self, server):
        """Simulate the read-only collection being populated by an external script."""
        await server.qdrant_connector.store(
            Entry(content="GET /users lists users", metadata={"method": "GET"}),
            collection_name="api_documentation",
        )
        return server

    async def test_find_works_on_read_only_collection(self, populated_server):
        found = await call(
            populated_server,
            "qdrant-find",
            {"query": "users", "collection_name": "api_documentation"},
        )
        assert "GET /users lists users" in found

    async def test_store_is_blocked(self, populated_server):
        result = await call(
            populated_server,
            "qdrant-store",
            {"information": "Injected", "collection_name": "api_documentation"},
        )

        assert result == "Error: collection 'api_documentation' is read-only."
        assert await contents_in(populated_server, "api_documentation") == [
            "GET /users lists users"
        ]

    async def test_edit_is_blocked(self, populated_server):
        result = await call(
            populated_server,
            "qdrant-edit",
            {
                "query": "users",
                "information": "Tampered",
                "collection_name": "api_documentation",
            },
        )

        assert result == "Error: collection 'api_documentation' is read-only."
        assert await contents_in(populated_server, "api_documentation") == [
            "GET /users lists users"
        ]

    async def test_delete_is_blocked(self, populated_server):
        result = await call(
            populated_server,
            "qdrant-delete",
            {"query": "users", "collection_name": "api_documentation"},
        )

        assert result == "Error: collection 'api_documentation' is read-only."
        assert await contents_in(populated_server, "api_documentation") == [
            "GET /users lists users"
        ]

    async def test_read_only_collection_does_not_block_writable_ones(
        self, populated_server
    ):
        result = await call(
            populated_server,
            "qdrant-store",
            {"information": "Still writable", "collection_name": "ai_thoughts"},
        )
        assert result == "Remembered: Still writable in collection ai_thoughts"


class TestUnknownCollectionsRejected:
    """Writes must only go to collections defined (and enabled) in the schemas."""

    EXPECTED_ERROR = (
        "Error: unknown collection '{name}'. "
        "Writable collections: ai_thoughts, project_notes."
    )

    @pytest.mark.parametrize("collection", ["ai_thougths", "archived"])
    async def test_store_is_rejected_and_no_collection_is_created(
        self, server, collection
    ):
        result = await call(
            server,
            "qdrant-store",
            {"information": "Lost memory", "collection_name": collection},
        )

        assert result == self.EXPECTED_ERROR.format(name=collection)
        assert await server.qdrant_connector.get_collection_names() == []

    @pytest.mark.parametrize("collection", ["ai_thougths", "archived"])
    async def test_edit_and_delete_are_rejected(self, server, collection):
        # Simulate data that already exists in Qdrant under that name
        await server.qdrant_connector.store(
            Entry(content="Existing"), collection_name=collection
        )

        edit = await call(
            server,
            "qdrant-edit",
            {"query": "x", "information": "Changed", "collection_name": collection},
        )
        delete = await call(
            server, "qdrant-delete", {"query": "x", "collection_name": collection}
        )

        assert edit == self.EXPECTED_ERROR.format(name=collection)
        assert delete == self.EXPECTED_ERROR.format(name=collection)
        assert await contents_in(server, collection) == ["Existing"]

    async def test_error_when_no_collection_is_writable(self, monkeypatch, make_server):
        monkeypatch.setenv(
            "METADATA_SCHEMAS", json.dumps({"docs": {"read_only": True}})
        )

        result = await call(
            make_server(),
            "qdrant-store",
            {"information": "x", "collection_name": "notes"},
        )

        assert (
            result == "Error: unknown collection 'notes'. Writable collections: none."
        )

    async def test_find_on_unknown_collection_returns_nothing(self, server):
        """Reads are harmless: they never create collections."""
        async with Client(FastMCPTransport(server)) as client:
            result = await client.call_tool(
                "qdrant-find", {"query": "x", "collection_name": "ai_thougths"}
            )

        assert result == []
        assert await server.qdrant_connector.get_collection_names() == []


# ──────────────────────────────────────────────
# Legacy single-collection mode (backward compatibility)
# ──────────────────────────────────────────────


@pytest.mark.usefixtures("legacy_mode")
class TestLegacySingleCollectionMode:
    async def test_tool_descriptions_use_defaults(self, server):
        tools = await server.get_tools()

        assert tools["qdrant-store"].description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert tools["qdrant-find"].description == DEFAULT_TOOL_FIND_DESCRIPTION
        assert tools["qdrant-edit"].description == DEFAULT_TOOL_EDIT_DESCRIPTION
        assert tools["qdrant-delete"].description == DEFAULT_TOOL_DELETE_DESCRIPTION

    async def test_collection_name_env_binds_single_collection(
        self, monkeypatch, make_server
    ):
        monkeypatch.setenv("COLLECTION_NAME", "memories")

        tools = await make_server().get_tools()

        for name in ALL_TOOLS:
            assert "collection_name" not in tools[name].parameters["properties"], name

    async def test_without_collection_name_env_parameter_is_required(self, server):
        tools = await server.get_tools()

        for name in ALL_TOOLS:
            assert "collection_name" in tools[name].parameters["required"], name

    async def test_any_collection_name_is_writable(self, server):
        result = await call(
            server,
            "qdrant-store",
            {"information": "Anything goes", "collection_name": "whatever"},
        )

        assert result == "Remembered: Anything goes in collection whatever"

    async def test_global_read_only_only_exposes_find_tool(
        self, monkeypatch, make_server
    ):
        monkeypatch.setenv("QDRANT_READ_ONLY", "1")

        tools = await make_server().get_tools()

        assert set(tools) == {"qdrant-find"}
