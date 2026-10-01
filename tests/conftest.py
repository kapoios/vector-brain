import copy
import json

import pytest

# Multi-collection schema used by the whole test suite. It mirrors the README examples:
# - two writable collections using the default "metadata" payload prefix
# - a read-only collection with top-level payload fields (populated externally)
# - a disabled collection that must be ignored everywhere
TEST_METADATA_SCHEMAS = {
    "ai_thoughts": {
        "enabled": True,
        "read_only": False,
        "field_prefix": "metadata",
        "fields": {
            "category": {
                "type": "keyword",
                "values": ["credential", "personal", "technical", "general"],
            },
            "tags": {"type": "keyword"},
            "importance": {
                "type": "keyword",
                "values": ["low", "medium", "high", "critical"],
            },
            "confidence": {"type": "float"},
        },
    },
    "project_notes": {
        "enabled": True,
        "read_only": False,
        "field_prefix": "metadata",
        "fields": {
            "project": {"type": "keyword"},
            "priority": {"type": "integer"},
        },
    },
    "api_documentation": {
        "enabled": True,
        "read_only": True,
        "field_prefix": "",
        "fields": {
            "type": {"type": "keyword", "values": ["endpoint", "data_model"]},
            "method": {
                "type": "keyword",
                "values": ["GET", "POST", "PUT", "DELETE", "PATCH"],
            },
            "path": {"type": "keyword"},
        },
    },
    "archived": {
        "enabled": False,
        "read_only": False,
        "fields": {"reason": {"type": "keyword"}},
    },
}

WRITABLE_COLLECTIONS = ["ai_thoughts", "project_notes"]
READ_ONLY_COLLECTIONS = ["api_documentation"]
ENABLED_COLLECTIONS = WRITABLE_COLLECTIONS + READ_ONLY_COLLECTIONS
DISABLED_COLLECTIONS = ["archived"]


@pytest.fixture
def metadata_schemas() -> dict:
    """A fresh copy of the multi-collection test schema."""
    return copy.deepcopy(TEST_METADATA_SCHEMAS)


@pytest.fixture(autouse=True)
def multi_collection_mode(monkeypatch, metadata_schemas):
    """
    Run every test in multi-collection mode by default.

    The schema is passed via METADATA_SCHEMAS (highest priority), so a developer's
    local metadata-schemas.json or shell environment cannot change test results.
    """
    monkeypatch.setenv("METADATA_SCHEMAS", json.dumps(metadata_schemas))
    for var in (
        "METADATA_SCHEMAS_FILE",
        "COLLECTION_NAME",
        "QDRANT_READ_ONLY",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "QDRANT_LOCAL_PATH",
        "QDRANT_ALLOW_ARBITRARY_FILTER",
        "TOOL_STORE_DESCRIPTION",
        "TOOL_FIND_DESCRIPTION",
        "TOOL_EDIT_DESCRIPTION",
        "TOOL_DELETE_DESCRIPTION",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def legacy_mode(monkeypatch, tmp_path):
    """
    Opt-in: no schemas configured, so the server falls back to the legacy
    single-collection behaviour (README: "Backward compatibility").
    """
    monkeypatch.delenv("METADATA_SCHEMAS", raising=False)
    # Avoid auto-discovering the project's real metadata-schemas.json
    monkeypatch.chdir(tmp_path)
