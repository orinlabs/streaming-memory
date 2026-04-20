"""
One-time ingest of memories into Supermemory under a stable container_tag.

Run once locally:
    uv run python scripts/ingest_supermemory.py \\
        --memories examples/dad_memories.json \\
        --container-tag family-assistant-prod

After this, any Modal deployment (or local script) that points a
SupermemoryPool at that container_tag can skip ingestion entirely and just
start querying.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from supermemory import Supermemory


def main() -> None:
    load_dotenv()

    p = argparse.ArgumentParser()
    p.add_argument("--memories", default="examples/dad_memories.json")
    p.add_argument("--container-tag", required=True)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument(
        "--wipe-first",
        action="store_true",
        help="Delete any existing docs under this tag before ingesting.",
    )
    args = p.parse_args()

    client = Supermemory()
    raw = json.loads(Path(args.memories).read_text())
    print(f"loaded {len(raw)} memories from {args.memories}")

    if args.wipe_first:
        resp = client.documents.delete_bulk(container_tags=[args.container_tag])
        print(f"wiped {getattr(resp, 'deleted_count', 0)} existing docs from {args.container_tag}")

    existing = client.documents.list(container_tags=[args.container_tag], limit=1)
    pag = getattr(existing, "pagination", None)
    existing_count = int(getattr(pag, "total_items", 0) or getattr(pag, "totalItems", 0) or 0)
    if existing_count and not args.wipe_first:
        print(
            f"container_tag {args.container_tag} already has {existing_count} docs — "
            "pass --wipe-first to re-ingest. aborting."
        )
        return

    t0 = time.time()
    ingested = 0
    for i in range(0, len(raw), args.batch_size):
        batch = raw[i : i + args.batch_size]
        docs = []
        for idx, m in enumerate(batch):
            docs.append(
                {
                    "content": m["content"],
                    "container_tag": args.container_tag,
                    "custom_id": f"mem_{i + idx}",
                    "metadata": {
                        "emotional_intensity": float(m.get("emotional_intensity", 0.5)),
                        "created_at": str(m.get("created_at", "")),
                        "local_idx": i + idx,
                    },
                    "task_type": "memory",
                }
            )
        client.documents.batch_add(documents=docs)
        ingested += len(batch)
        print(f"  batched {ingested}/{len(raw)}  ({time.time() - t0:.1f}s)")

    print(f"done — ingested {ingested} memories into {args.container_tag} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
