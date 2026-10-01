# 🧠 vector-brain

**Give your AI a long-term memory that actually stays organised.**

Most LLM chats forget everything the moment the window closes. vector-brain is an
[MCP](https://modelcontextprotocol.io) server that gives any MCP-capable assistant a persistent,
searchable brain backed by the [Qdrant](https://qdrant.tech) vector database. The assistant can
remember facts, look them up by meaning, fix them when they change, and forget them when you ask.

What sets it apart is that the brain has **regions**. Instead of one big pile of memories, you set
up separate collections, each with its own structure and rules. Personal facts, project notes and
reference docs each live in their own collection, and the assistant is told exactly how to use
each one.

```
                  +------------------------ vector-brain -------------------------+
 "Remember I      |                                                               |
  prefer dark  -->|  ai_thoughts         project_notes       api_documentation    |
  mode"           |  read + write        read + write        read only            |
                  |  category, tags,     project, priority   method, path, type   |
 "What do you     |  importance, ...                                              |
  know about   -->|                                                               |
  my setup?"      |      search by meaning  +  exact metadata filters             |
                  +-------------------------------+-------------------------------+
                                                  |
                                           Qdrant vector DB
```

## Why vector-brain?

- **🗂️ Memory with regions.** Define as many collections as you like in one
  `metadata-schemas.json`. Each one is a separate area of the brain with its own fields.
- **🏷️ Structured recall, not just similarity.** Every memory carries typed metadata
  (`category`, `importance`, `confidence`, `tags`, anything you define). The assistant can
  combine meaning-based search with exact filters, for example *"credentials with high
  importance"*.
- **🗣️ Self-describing tools.** Tool descriptions are generated from your schemas, so the
  assistant knows which collections exist, which fields each one has and which values are
  allowed, without any prompt engineering on your part.
- **📖 Knowledge it can read but not change.** Mark a collection `read_only` (for example API
  docs filled by an ingestion script). The assistant can search it but can never overwrite or
  delete it.
- **🛡️ No stray memories.** Writes are only accepted into collections you defined and enabled. A
  mistyped collection name returns an error instead of quietly creating a new collection.
- **🕰️ It knows when it learned something.** `created_at` is stamped on every new memory and
  `updated_at` on every edit, and both are indexed so they can be filtered.
- **✏️ Memories can change.** Beyond store and search, the assistant can edit or delete a memory
  by describing it. No IDs to track.
- **⚡ Fast filtering.** Qdrant payload indexes are created automatically for every schema field.
- **🔌 Works with your client.** stdio, SSE or streamable HTTP, with local embeddings via
  FastEmbed, so no API key is needed for embeddings.

## Quick Start

**Prerequisites:** Python 3.10+, [uv](https://docs.astral.sh/uv/) and a running Qdrant instance
(for example `docker run -p 6333:6333 qdrant/qdrant`).

### 1. Install

```bash
git clone https://github.com/kapoios/vector-brain.git
cd vector-brain
uv sync
```

### 2. Design your brain

Create `metadata-schemas.json` in the server's working directory (the repo ships with an example):

```json
{
  "ai_thoughts": {
    "enabled": true,
    "read_only": false,
    "field_prefix": "metadata",
    "fields": {
      "category": {
        "type": "keyword",
        "values": ["credential", "personal", "technical", "social", "financial", "health", "work", "general"]
      },
      "tags": { "type": "keyword" },
      "subject": { "type": "keyword" },
      "importance": { "type": "keyword", "values": ["low", "medium", "high", "critical"] },
      "confidence": { "type": "float" },
      "source": { "type": "keyword", "values": ["user_stated", "observed", "inferred", "researched"] },
      "sentiment": { "type": "keyword", "values": ["positive", "negative", "neutral", "urgent"] }
    }
  },
  "api_documentation": {
    "enabled": true,
    "read_only": true,
    "field_prefix": "",
    "fields": {
      "type": { "type": "keyword", "values": ["endpoint", "data_model"] },
      "method": { "type": "keyword", "values": ["GET", "POST", "PUT", "DELETE", "PATCH"] },
      "path": { "type": "keyword" },
      "tags": { "type": "keyword" }
    }
  }
}
```

### 3. Connect your assistant

Add it to your MCP client (for example LM Studio's `mcp.json`, Claude Desktop or Cursor):

```json
{
  "vector-brain": {
    "command": "uv",
    "args": ["run", "vector-brain"],
    "cwd": "C:\\path\\to\\vector-brain",
    "env": {
      "QDRANT_URL": "http://127.0.0.1:6333",
      "EMBEDDING_MODEL": "jinaai/jina-embeddings-v3",
      "QDRANT_ALLOW_ARBITRARY_FILTER": "true"
    }
  }
}
```
#### Assistants without a `cwd` setting (e.g. Unsloth desktop)

Some MCP clients, such as the Unsloth desktop app, only let you set an executable, its arguments
and environment variables. They have no working-directory (`cwd`) field. That breaks the plain
setup in two ways:

- `uv run vector-brain` is started from whatever folder the app happens to be in, so uv can't
  find the project and fails with `program not found`.
- vector-brain looks for `metadata-schemas.json` in the current folder, so even if it started,
  your collections wouldn't load.

The fix is to use `uv` as the executable with the arguments `run`, `--directory`, `C:\path\to\vector-brain`, `vector-brain`.


**That's it**. On startup vector-brain finds `metadata-schemas.json` in `cwd` and:

- generates tool descriptions listing every collection, its fields and allowed values
- creates Qdrant payload indexes for each collection
- enforces `read_only` per collection
- rejects writes to collections that aren't defined (or are disabled)
- adds `created_at` / `updated_at` timestamps automatically

To use a network transport instead of stdio, run `uv run vector-brain --transport sse` or
`--transport streamable-http`.


## The memory lifecycle

| The assistant wants to… | Tool | What happens |
|---|---|---|
| **Remember** something | `qdrant-store` | The text is embedded and saved with its metadata and a `created_at` timestamp |
| **Recall** something | `qdrant-find` | Search by meaning in any collection, optionally narrowed with metadata filters |
| **Update** a memory | `qdrant-edit` | The closest match is rewritten in place; metadata is kept unless replaced; `updated_at` is set |
| **Forget** a memory | `qdrant-delete` | The closest match is removed and its content is returned as confirmation |

A typical conversation:

> **You:** My staging server is at 10.0.0.12, remember that.
> *→ `qdrant-store` into `ai_thoughts` with `{"category": "technical", "importance": "high"}`*
>
> **You (a week later):** What's the staging IP again?
> *→ `qdrant-find` in `ai_thoughts` → "10.0.0.12"*
>
> **You:** We moved staging to 10.0.0.40.
> *→ `qdrant-edit` rewrites the memory, keeps its metadata, stamps `updated_at`*

## Tools Reference

### `qdrant-store`
Store a memory with metadata. Blocked on read-only and unconfigured collections. Adds a
`created_at` timestamp unless one is supplied.
- `information` (string): text to remember
- `collection_name` (string): target collection (only writable collections are listed)
- `metadata` (JSON, optional): fields as defined in the collection's schema

### `qdrant-find`
Search any enabled collection, including read-only ones.
- `query` (string): what to search for, matched by meaning
- `collection_name` (string): collection to search (all enabled collections are listed)
- `query_filter` (JSON, optional, needs `QDRANT_ALLOW_ARBITRARY_FILTER=true`): a Qdrant filter, e.g.
  `{"must": [{"key": "metadata.category", "match": {"value": "credential"}}]}`

Returns the matching memories with their content and metadata.

### `qdrant-edit`
Rewrite the closest matching memory. Blocked on read-only and unconfigured collections. Existing
metadata is kept unless new metadata is given, and `updated_at` is always set.
- `query` (string): describes the memory to update
- `information` (string): the replacement text
- `collection_name` (string): only writable collections are listed
- `metadata` (JSON, optional): replacement metadata
- `query_filter` (JSON, optional, needs `QDRANT_ALLOW_ARBITRARY_FILTER=true`): Qdrant filter to
  narrow which memory is matched

### `qdrant-delete`
Forget the closest matching memory. Blocked on read-only and unconfigured collections.
- `query` (string): describes the memory to delete
- `collection_name` (string): only writable collections are listed

## Schema Reference

### `metadata-schemas.json` format

```json
{
  "collection_name": {
    "enabled": true,
    "read_only": false,
    "field_prefix": "metadata",
    "fields": {
      "field_name": {
        "type": "keyword|float|integer|boolean",
        "values": ["optional", "list", "of", "allowed", "values"]
      }
    }
  }
}
```

| Property | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `true` | Set to `false` to switch a collection off without deleting its config (writes to it are rejected) |
| `read_only` | boolean | `false` | If `true`, store, edit and delete are blocked for this collection |
| `field_prefix` | string | `"metadata"` | Payload path prefix. Use `""` for top-level fields (e.g. externally populated collections) |
| `fields` | object | `{}` | Field definitions. Each field has a `type` and optional `values` |
| `fields.*.type` | string | `"keyword"` | One of `keyword`, `float`, `integer`, `boolean` |
| `fields.*.values` | array | `null` | Allowed values, shown to the assistant in the tool description |

### Where schemas are loaded from

1. `METADATA_SCHEMAS` env var (inline JSON string), highest priority
2. `METADATA_SCHEMAS_FILE` env var (path to a JSON file)
3. `metadata-schemas.json` in `cwd`, found automatically (recommended)

### Single-collection mode

If no schemas are found, vector-brain runs as a simple single-collection memory:
- `COLLECTION_NAME` fixes the collection used by every tool
- tool descriptions come from the defaults or the `TOOL_*_DESCRIPTION` env vars

In either mode, `QDRANT_READ_ONLY=true` makes the whole server read-only: only `qdrant-find` is
exposed.

## Example: a personal brain plus read-only reference docs

```json
{
  "ai_thoughts": {
    "read_only": false,
    "fields": {
      "category": { "type": "keyword", "values": ["credential", "personal", "technical"] },
      "tags": { "type": "keyword" },
      "importance": { "type": "keyword", "values": ["low", "medium", "high", "critical"] }
    }
  },
  "api_documentation": {
    "read_only": true,
    "field_prefix": "",
    "fields": {
      "type": { "type": "keyword", "values": ["endpoint", "data_model"] },
      "method": { "type": "keyword" },
      "path": { "type": "keyword" }
    }
  }
}
```

The assistant builds up its own memories in `ai_thoughts` and looks things up in
`api_documentation`, which is filled by an external process such as an OpenAPI ingestion script.
It can search the docs but never change them.

## Environment Variables

| Name | Description | Default |
|---|---|---|
| `QDRANT_URL` | URL of the Qdrant server | None |
| `QDRANT_API_KEY` | API key for the Qdrant server | None |
| `QDRANT_LOCAL_PATH` | Path to a local on-disk Qdrant database (instead of `QDRANT_URL`) | None |
| `EMBEDDING_PROVIDER` | Embedding provider (`fastembed`) | `fastembed` |
| `EMBEDDING_MODEL` | Embedding model name | `sentence-transformers/all-MiniLM-L6-v2` |
| `QDRANT_ALLOW_ARBITRARY_FILTER` | Let the assistant build metadata filters | `false` |
| `QDRANT_SEARCH_LIMIT` | Max results per search | `10` |
| `METADATA_SCHEMAS` | Inline JSON schemas (overrides the file) | None |
| `METADATA_SCHEMAS_FILE` | Path to a schemas JSON file | None |
| `COLLECTION_NAME` | Fixed collection (single-collection mode only) | None |
| `QDRANT_READ_ONLY` | Global read-only flag: hides store, edit and delete | `false` |
| `TOOL_STORE_DESCRIPTION` | Override the generated store description | Generated |
| `TOOL_FIND_DESCRIPTION` | Override the generated find description | Generated |
| `TOOL_EDIT_DESCRIPTION` | Override the generated edit description | Generated |
| `TOOL_DELETE_DESCRIPTION` | Override the generated delete description | Generated |

## Development

```bash
uv sync           # install with dev dependencies
uv run pytest     # run the test suite (uses in-memory Qdrant, no server needed)
```
---
Parts of this project — including code and documentation — were developed with the
assistance of Cline (AI coding agent) using Anthropic's Claude models, under human direction and review.
