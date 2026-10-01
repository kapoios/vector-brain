import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings

from vector_brain.embeddings.types import EmbeddingProviderType

METADATA_PATH = "metadata"


# ──────────────────────────────────────────────
# Per-collection metadata schema models
# ──────────────────────────────────────────────


class MetadataFieldDef(BaseModel):
    """Definition of a single metadata field."""

    type: Literal["keyword", "float", "integer", "boolean"] = "keyword"
    values: list[str] | None = None  # Optional list of allowed values


class CollectionSchema(BaseModel):
    """Schema definition for a single collection."""

    enabled: bool = True  # Set to false to disable without removing from config
    read_only: bool = False
    field_prefix: str = "metadata"  # "" for top-level payload fields
    fields: dict[str, MetadataFieldDef] = {}


# ──────────────────────────────────────────────
# Auto-generated tool descriptions from schemas
# ──────────────────────────────────────────────


def generate_store_description(schemas: dict[str, CollectionSchema]) -> str:
    """Only mention writable collections. Don't mention read-only ones at all."""
    writable = {n: s for n, s in schemas.items() if not s.read_only}
    if not writable:
        return "Store a memory. Always include metadata."
    parts = ["Store a memory. Always include metadata."]
    for name, schema in writable.items():
        field_parts = []
        for fname, fdef in schema.fields.items():
            if fdef.values:
                field_parts.append(f"{fname} ({'|'.join(fdef.values)})")
            else:
                field_parts.append(fname)
        parts.append(f"collection_name={name}: {', '.join(field_parts)}")
    return "\n".join(parts)


def generate_find_description(schemas: dict[str, CollectionSchema]) -> str:
    if not schemas:
        return "Search memories."
    parts = ["Search memories."]
    for name, schema in schemas.items():
        ro = " (search only)" if schema.read_only else ""
        prefix = f"{schema.field_prefix}." if schema.field_prefix else ""
        field_names = [f"{prefix}{f}" for f in schema.fields.keys()]
        parts.append(f"collection_name={name}{ro}: filter by {', '.join(field_names)}")
    parts.append('Filter: {"must": [{"key": "FIELD", "match": {"value": "..."}}]}')
    return "\n".join(parts)


def generate_delete_description(schemas: dict[str, CollectionSchema]) -> str:
    """Only mention writable collections. Don't mention read-only ones at all."""
    writable = [n for n, s in schemas.items() if not s.read_only]
    if writable:
        return f"Delete a memory by semantic search. collection_name must be: {', '.join(writable)}"
    return "Delete a memory by semantic search."


def generate_edit_description(schemas: dict[str, CollectionSchema]) -> str:
    """Only mention writable collections. Don't mention read-only ones at all."""
    writable = [n for n, s in schemas.items() if not s.read_only]
    if writable:
        return (
            f"Update an existing memory. collection_name must be: {', '.join(writable)}"
        )
    return "Update an existing memory."


# ──────────────────────────────────────────────
# Fallback defaults (when METADATA_SCHEMAS is not set)
# ──────────────────────────────────────────────

DEFAULT_TOOL_STORE_DESCRIPTION = "Store a memory. Always include metadata."
DEFAULT_TOOL_DELETE_DESCRIPTION = "Delete a memory by semantic search."
DEFAULT_TOOL_EDIT_DESCRIPTION = "Update an existing memory."
DEFAULT_TOOL_FIND_DESCRIPTION = "Search memories."


# ──────────────────────────────────────────────
# Settings classes
# ──────────────────────────────────────────────


class ToolSettings(BaseSettings):
    """Configuration for all the tools."""

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_edit_description: str = Field(
        default=DEFAULT_TOOL_EDIT_DESCRIPTION,
        validation_alias="TOOL_EDIT_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )
    tool_delete_description: str = Field(
        default=DEFAULT_TOOL_DELETE_DESCRIPTION,
        validation_alias="TOOL_DELETE_DESCRIPTION",
    )


class EmbeddingProviderSettings(BaseSettings):
    """Configuration for the embedding provider."""

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )


class FilterableField(BaseModel):
    name: str = Field(description="The name of the field payload field to filter on")
    description: str = Field(
        description="A description for the field used in the tool description"
    )
    field_type: Literal["keyword", "integer", "float", "boolean"] = Field(
        description="The type of the field"
    )
    condition: Literal["==", "!=", ">", ">=", "<", "<=", "any", "except"] | None = (
        Field(
            default=None,
            description=(
                "The condition to use for the filter. If not provided, the field will be indexed, but no "
                "filter argument will be exposed to MCP tool."
            ),
        )
    )
    required: bool = Field(
        default=False,
        description="Whether the field is required for the filter.",
    )


class QdrantSettings(BaseSettings):
    """Configuration for the Qdrant connector."""

    location: str | None = Field(default=None, validation_alias="QDRANT_URL")
    api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str | None = Field(
        default=None, validation_alias="COLLECTION_NAME"
    )
    local_path: str | None = Field(default=None, validation_alias="QDRANT_LOCAL_PATH")
    search_limit: int = Field(default=10, validation_alias="QDRANT_SEARCH_LIMIT")
    read_only: bool = Field(default=False, validation_alias="QDRANT_READ_ONLY")

    # Per-collection metadata schemas
    # Priority: METADATA_SCHEMAS (inline JSON) > METADATA_SCHEMAS_FILE > auto-discover metadata-schemas.json in cwd
    metadata_schemas_raw: str | None = Field(
        default=None, validation_alias="METADATA_SCHEMAS"
    )
    metadata_schemas_file: str | None = Field(
        default=None, validation_alias="METADATA_SCHEMAS_FILE"
    )

    filterable_fields: list[FilterableField] | None = Field(default=None)

    allow_arbitrary_filter: bool = Field(
        default=False, validation_alias="QDRANT_ALLOW_ARBITRARY_FILTER"
    )

    @property
    def metadata_schemas(self) -> dict[str, CollectionSchema]:
        """Load metadata schemas. Priority: inline JSON > file path > auto-discover in cwd."""
        raw_json = None

        # 1. Inline JSON string
        if self.metadata_schemas_raw:
            raw_json = self.metadata_schemas_raw
        # 2. Explicit file path
        elif self.metadata_schemas_file:
            raw_json = Path(self.metadata_schemas_file).read_text(encoding="utf-8")
        # 3. Auto-discover metadata-schemas.json in cwd
        else:
            auto_path = Path(os.getcwd()) / "metadata-schemas.json"
            if auto_path.exists():
                raw_json = auto_path.read_text(encoding="utf-8")

        if raw_json is None:
            return {}
        raw = json.loads(raw_json)
        # Filter out disabled collections
        return {
            name: CollectionSchema(**schema)
            for name, schema in raw.items()
            if schema.get("enabled", True)
        }

    @property
    def all_collection_names(self) -> list[str]:
        return list(self.metadata_schemas.keys())

    @property
    def writable_collection_names(self) -> list[str]:
        return [n for n, s in self.metadata_schemas.items() if not s.read_only]

    def filterable_fields_dict(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {field.name: field for field in self.filterable_fields}

    def filterable_fields_dict_with_conditions(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {
            field.name: field
            for field in self.filterable_fields
            if field.condition is not None
        }

    @model_validator(mode="after")
    def check_local_path_conflict(self) -> "QdrantSettings":
        if self.local_path:
            if self.location is not None or self.api_key is not None:
                raise ValueError(
                    "If 'local_path' is set, 'location' and 'api_key' must be None."
                )
        return self
