import json
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient, models

from tests.conftest import ENABLED_COLLECTIONS, WRITABLE_COLLECTIONS
from vector_brain.embeddings.base import EmbeddingProvider
from vector_brain.qdrant import Entry, QdrantConnector, build_indexes_for_collection
from vector_brain.settings import (
    CollectionSchema,
    MetadataFieldDef,
    QdrantSettings,
    generate_delete_description,
    generate_edit_description,
    generate_find_description,
    generate_store_description,
)

KEYWORD = models.PayloadSchemaType.KEYWORD


# ──────────────────────────────────────────────
# Schema loading
# ──────────────────────────────────────────────


class TestSchemaLoading:
    def test_inline_json_is_parsed(self):
        schemas = QdrantSettings().metadata_schemas

        assert list(schemas) == ENABLED_COLLECTIONS
        assert all(isinstance(s, CollectionSchema) for s in schemas.values())
        assert schemas["api_documentation"].read_only is True
        assert schemas["api_documentation"].field_prefix == ""
        assert schemas["ai_thoughts"].fields["confidence"].type == "float"

    def test_disabled_collections_are_dropped(self):
        settings = QdrantSettings()

        assert "archived" not in settings.metadata_schemas
        assert "archived" not in settings.all_collection_names

    def test_collection_name_helpers(self):
        settings = QdrantSettings()

        assert settings.all_collection_names == ENABLED_COLLECTIONS
        assert settings.writable_collection_names == WRITABLE_COLLECTIONS

    def test_schema_defaults(self, monkeypatch):
        monkeypatch.setenv("METADATA_SCHEMAS", json.dumps({"notes": {}}))

        schema = QdrantSettings().metadata_schemas["notes"]

        assert schema.enabled is True
        assert schema.read_only is False
        assert schema.field_prefix == "metadata"
        assert schema.fields == {}

    def test_schemas_file(self, monkeypatch, tmp_path):
        path = tmp_path / "custom-schemas.json"
        path.write_text(json.dumps({"from_file": {"fields": {}}}), encoding="utf-8")
        monkeypatch.delenv("METADATA_SCHEMAS")
        monkeypatch.setenv("METADATA_SCHEMAS_FILE", str(path))

        assert QdrantSettings().all_collection_names == ["from_file"]

    def test_auto_discovers_schemas_file_in_cwd(self, monkeypatch, tmp_path):
        (tmp_path / "metadata-schemas.json").write_text(
            json.dumps({"discovered": {"fields": {}}}), encoding="utf-8"
        )
        monkeypatch.delenv("METADATA_SCHEMAS")
        monkeypatch.chdir(tmp_path)

        assert QdrantSettings().all_collection_names == ["discovered"]

    def test_priority_inline_over_file_over_auto_discovery(self, monkeypatch, tmp_path):
        (tmp_path / "metadata-schemas.json").write_text(
            json.dumps({"discovered": {}}), encoding="utf-8"
        )
        explicit = tmp_path / "explicit.json"
        explicit.write_text(json.dumps({"from_file": {}}), encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("METADATA_SCHEMAS_FILE", str(explicit))
        monkeypatch.setenv("METADATA_SCHEMAS", json.dumps({"inline": {}}))

        assert QdrantSettings().all_collection_names == ["inline"]

        monkeypatch.delenv("METADATA_SCHEMAS")
        assert QdrantSettings().all_collection_names == ["from_file"]

        monkeypatch.delenv("METADATA_SCHEMAS_FILE")
        assert QdrantSettings().all_collection_names == ["discovered"]

    def test_no_schemas_means_legacy_mode(self, legacy_mode):
        assert QdrantSettings().metadata_schemas == {}

    def test_project_schema_file_is_valid(self, monkeypatch):
        """The metadata-schemas.json shipped in the repo root must load cleanly."""
        project_file = Path(__file__).resolve().parent.parent / "metadata-schemas.json"
        if not project_file.exists():
            pytest.skip("No metadata-schemas.json in project root")
        monkeypatch.delenv("METADATA_SCHEMAS")
        monkeypatch.setenv("METADATA_SCHEMAS_FILE", str(project_file))

        assert len(QdrantSettings().metadata_schemas) > 0

    def test_invalid_field_type_is_rejected(self, monkeypatch):
        monkeypatch.setenv(
            "METADATA_SCHEMAS",
            json.dumps({"bad": {"fields": {"x": {"type": "datetime"}}}}),
        )

        with pytest.raises(ValueError):
            _ = QdrantSettings().metadata_schemas


# ──────────────────────────────────────────────
# Description generators
# ──────────────────────────────────────────────


class TestDescriptionGenerators:
    def test_only_read_only_collections(self):
        schemas = {"docs": CollectionSchema(read_only=True)}

        assert (
            generate_store_description(schemas)
            == "Store a memory. Always include metadata."
        )
        assert (
            generate_delete_description(schemas)
            == "Delete a memory by semantic search."
        )
        assert generate_edit_description(schemas) == "Update an existing memory."
        assert "collection_name=docs (search only)" in generate_find_description(
            schemas
        )

    def test_find_description_includes_filter_syntax(self):
        description = generate_find_description(QdrantSettings().metadata_schemas)

        assert description.splitlines()[-1] == (
            'Filter: {"must": [{"key": "FIELD", "match": {"value": "..."}}]}'
        )


# ──────────────────────────────────────────────
# Per-collection payload indexes
# ──────────────────────────────────────────────


class TestBuildIndexes:
    def test_writable_collection_gets_field_and_timestamp_indexes(self):
        schema = QdrantSettings().metadata_schemas["ai_thoughts"]

        assert build_indexes_for_collection(schema) == {
            "metadata.category": KEYWORD,
            "metadata.tags": KEYWORD,
            "metadata.importance": KEYWORD,
            "metadata.confidence": models.PayloadSchemaType.FLOAT,
            "metadata.created_at": KEYWORD,
            "metadata.updated_at": KEYWORD,
        }

    def test_integer_and_boolean_types(self):
        schema = CollectionSchema(
            read_only=True,
            fields={
                "priority": MetadataFieldDef(type="integer"),
                "done": MetadataFieldDef(type="boolean"),
            },
        )

        assert build_indexes_for_collection(schema) == {
            "metadata.priority": models.PayloadSchemaType.INTEGER,
            "metadata.done": models.PayloadSchemaType.BOOL,
        }

    def test_read_only_top_level_collection_has_no_prefix_or_timestamps(self):
        schema = QdrantSettings().metadata_schemas["api_documentation"]

        assert build_indexes_for_collection(schema) == {
            "type": KEYWORD,
            "method": KEYWORD,
            "path": KEYWORD,
        }

    def test_unknown_collection_has_no_indexes(self):
        assert build_indexes_for_collection(None) == {}


# ──────────────────────────────────────────────
# Connector with multiple collections
# ──────────────────────────────────────────────


class FakeEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        return [1.0, 0.0]

    def get_vector_name(self) -> str:
        return "fake"

    def get_vector_size(self) -> int:
        return 2


IndexCall = tuple[str, str, models.PayloadSchemaType]


@pytest.fixture
def index_calls() -> list[IndexCall]:
    """Payload index requests made by the connector, as (collection, field, type)."""
    return []


@pytest.fixture
def connector(monkeypatch, index_calls) -> QdrantConnector:
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=None,
        embedding_provider=FakeEmbeddingProvider(),
        field_indexes={"legacy_field": KEYWORD},
        metadata_schemas=QdrantSettings().metadata_schemas,
    )

    # Local Qdrant ignores payload indexes (and warns), so record the requests instead.
    async def record_index(_self, collection_name, field_name, field_schema, **_):
        index_calls.append((collection_name, field_name, field_schema))

    monkeypatch.setattr(AsyncQdrantClient, "create_payload_index", record_index)
    return connector


def indexed_fields(index_calls: list[IndexCall], collection: str) -> dict:
    return {f: t for c, f, t in index_calls if c == collection}


class TestConnectorMultiCollection:
    async def test_each_collection_gets_its_own_schema_indexes(
        self, connector, index_calls
    ):
        await connector.store(Entry(content="a"), collection_name="ai_thoughts")
        await connector.store(Entry(content="b"), collection_name="project_notes")

        thoughts = indexed_fields(index_calls, "ai_thoughts")
        notes = indexed_fields(index_calls, "project_notes")

        assert "metadata.category" in thoughts
        assert "metadata.project" not in thoughts
        assert notes["metadata.priority"] == models.PayloadSchemaType.INTEGER
        assert "metadata.category" not in notes

    async def test_legacy_indexes_are_merged_in(self, connector, index_calls):
        await connector.store(Entry(content="a"), collection_name="ai_thoughts")

        assert indexed_fields(index_calls, "ai_thoughts")["legacy_field"] == KEYWORD

    async def test_unknown_collection_only_gets_legacy_indexes(
        self, connector, index_calls
    ):
        await connector.store(Entry(content="a"), collection_name="scratch")

        assert indexed_fields(index_calls, "scratch") == {"legacy_field": KEYWORD}

    async def test_indexes_are_created_once_per_collection(
        self, connector, index_calls
    ):
        await connector.store(Entry(content="a"), collection_name="ai_thoughts")
        first = len(index_calls)
        await connector.store(Entry(content="b"), collection_name="ai_thoughts")

        assert len(index_calls) == first

    async def test_collections_are_created_on_demand(self, connector):
        await connector.store(Entry(content="a"), collection_name="ai_thoughts")
        await connector.store(Entry(content="b"), collection_name="project_notes")

        assert sorted(await connector.get_collection_names()) == sorted(
            WRITABLE_COLLECTIONS
        )

    async def test_search_edit_delete_are_scoped_to_collection(self, connector):
        await connector.store(Entry(content="thought"), collection_name="ai_thoughts")
        await connector.store(Entry(content="note"), collection_name="project_notes")

        edited = await connector.edit(
            "x", Entry(content="edited note"), collection_name="project_notes"
        )
        assert edited is not None
        deleted = await connector.delete("x", collection_name="ai_thoughts")
        assert [e.content for e in deleted] == ["thought"]

        assert await connector.search("x", collection_name="ai_thoughts") == []
        notes = await connector.search("x", collection_name="project_notes")
        assert [e.content for e in notes] == ["edited note"]

    async def test_missing_collection_returns_empty(self, connector):
        assert await connector.search("x", collection_name="ai_thoughts") == []
        assert await connector.delete("x", collection_name="ai_thoughts") == []
        assert (
            await connector.edit("x", Entry(content="y"), collection_name="ai_thoughts")
            is None
        )
