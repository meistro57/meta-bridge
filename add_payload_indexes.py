# add_payload_indexes.py
from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


load_dotenv()

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip() or None

INDEXES: dict[str, list[tuple[str, Any]]] = {
    "vectoreology_findings": [
        ("type",       models.PayloadSchemaType.KEYWORD),
        ("subject",    models.PayloadSchemaType.KEYWORD),
        ("is_anomaly", models.PayloadSchemaType.BOOL),
        ("confidence", models.PayloadSchemaType.FLOAT),
        ("clusters",   models.PayloadSchemaType.KEYWORD),
        ("stored_at",  models.PayloadSchemaType.KEYWORD),
    ],
    "misfit_reports": [
        ("source_file", models.PayloadSchemaType.KEYWORD),
        ("mined_at",    models.PayloadSchemaType.KEYWORD),
    ],
    "mb_claims": [
        ("attributions[].source_id", models.PayloadSchemaType.KEYWORD),
        ("tags[]", models.PayloadSchemaType.KEYWORD),
        ("entity_type", models.PayloadSchemaType.KEYWORD),
        ("editorial_status", models.PayloadSchemaType.KEYWORD),
        ("chapter", models.PayloadSchemaType.KEYWORD),
        (
            "canonical_statement",
            models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
            ),
        ),
    ],
    "mb_chunks": [
        ("source_id", models.PayloadSchemaType.KEYWORD),
        ("chapter", models.PayloadSchemaType.KEYWORD),
    ],
    "mb_sources": [
        ("id",           models.PayloadSchemaType.KEYWORD),
        ("title",        models.PayloadSchemaType.KEYWORD),
        ("tradition",    models.PayloadSchemaType.KEYWORD),
        ("channel_type", models.PayloadSchemaType.KEYWORD),
        ("entity_type",  models.PayloadSchemaType.KEYWORD),
        ("author",       models.PayloadSchemaType.KEYWORD),
    ],
}


def _to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _to_plain(value.model_dump())
    if hasattr(value, "dict"):
        return _to_plain(value.dict())
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_plain(v) for v in value)
    if hasattr(value, "value"):
        return value.value
    return value


def _schema_type_name(field_schema: Any) -> str:
    if isinstance(field_schema, models.PayloadSchemaType):
        return field_schema.value
    if hasattr(field_schema, "type") and field_schema.type:
        value = field_schema.type
        return value.value if hasattr(value, "value") else str(value)
    return str(field_schema)


def _is_index_already_matching(existing: Any, desired_schema: Any) -> bool:
    if existing is None:
        return False

    existing_data_type = getattr(existing, "data_type", None)
    existing_params = getattr(existing, "params", None)

    if isinstance(desired_schema, models.PayloadSchemaType):
        desired = desired_schema.value
        current = existing_data_type.value if hasattr(existing_data_type, "value") else str(existing_data_type)
        return current == desired

    desired_type = _schema_type_name(desired_schema)
    current_type = existing_data_type.value if hasattr(existing_data_type, "value") else str(existing_data_type)
    if current_type != desired_type:
        return False

    desired_tokenizer = getattr(desired_schema, "tokenizer", None)
    if desired_tokenizer is None:
        return True

    existing_tokenizer = getattr(existing_params, "tokenizer", None)
    desired_tokenizer_name = desired_tokenizer.value if hasattr(desired_tokenizer, "value") else str(desired_tokenizer)
    existing_tokenizer_name = (
        existing_tokenizer.value if hasattr(existing_tokenizer, "value") else str(existing_tokenizer)
    )
    return desired_tokenizer_name == existing_tokenizer_name


def ensure_indexes(client: QdrantClient, collection_name: str, index_defs: list[tuple[str, Any]]) -> None:
    payload_schema = client.get_collection(collection_name=collection_name).payload_schema or {}

    print(f"\n[{collection_name}] ensuring indexes")
    for field_name, field_schema in index_defs:
        existing = payload_schema.get(field_name)
        if _is_index_already_matching(existing, field_schema):
            print(f"- {field_name}: already indexed ({_schema_type_name(field_schema)})")
            continue

        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=True,
        )
        print(f"- {field_name}: indexed ({_schema_type_name(field_schema)})")


def print_payload_schema(client: QdrantClient, collection_name: str) -> None:
    payload_schema = client.get_collection(collection_name=collection_name).payload_schema or {}
    rendered = _to_plain(payload_schema)
    print(f"\n[{collection_name}] payload schema")
    print(json.dumps(rendered, indent=2, sort_keys=True, default=str))


def main() -> None:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    for collection_name, index_defs in INDEXES.items():
        ensure_indexes(client, collection_name, index_defs)

    for collection_name in INDEXES:
        print_payload_schema(client, collection_name)


if __name__ == "__main__":
    main()
