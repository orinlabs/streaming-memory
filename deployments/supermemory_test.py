"""
Run the streaming memory engine on Modal with SupermemoryPool swapped in for
the in-process Hebbian MemoryPool.

The existing StreamingMemoryService is used unchanged — every ``update_every_n``
tokens it takes the last ``lookback_tokens`` of the reasoning trace and calls
``pool.retrieve(...)``. Here that pool is backed by Supermemory's hosted
memory API instead of local BGE embeddings.

Run:
    modal run deployments/supermemory_test.py \\
        --question "Dad's birthday is next week. What should I get him?" \\
        --limit-ingest 30 --update-every-n 10 --max-memories 5

Requires a local .env with SUPERMEMORY_API_KEY set (pulled in via
modal.Secret.from_dotenv()).
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import modal

APP_NAME = "supermemory-streaming-test"
MODEL_ID = "Qwen/Qwen3-8B"

repo_root = Path(__file__).parent.parent
package_path = repo_root / "streaming_memory"
dad_memories_path = repo_root / "examples" / "dad_memories.json"


def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, ignore_patterns=["*.gguf"])
    print(f"Downloaded {MODEL_ID}")


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "accelerate>=0.28",
        "numpy",
        "huggingface_hub",
        "pydantic",
        "sentencepiece>=0.1.99",
        "openai",
        "deepgram-sdk>=3.0",
        "websockets",
        "supermemory>=3.34.0",
        pre=True,
    )
    .run_function(download_model)
    .add_local_dir(package_path, "/root/streaming_memory")
    .add_local_file(dad_memories_path, "/app/dad_memories.json")
)


@app.cls(
    image=image,
    gpu="A100",
    timeout=900,
    scaledown_window=120,
    secrets=[modal.Secret.from_dotenv(str(repo_root / ".env"))],
)
class SupermemoryStreamingTester:
    """Runs StreamingMemoryService with a Supermemory-backed pool."""

    @modal.enter()
    def startup(self):
        import sys

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        sys.path.insert(0, "/root")

        print(f"🚀 Loading {MODEL_ID}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"✅ LLM loaded on {next(self.model.parameters()).device}")

    @modal.method()
    def run_streaming(
        self,
        question: str,
        memories: list[dict],
        container_tag: str,
        system_prompt: str,
        update_every_n: int = 10,
        max_memories: int = 5,
        lookback_tokens: int = 60,
        max_output_tokens: int = 1500,
        cleanup: bool = True,
    ):
        """Ingest memories into Supermemory, then stream generation.

        Yields the same StreamEvent payloads (type + data dict) that
        StreamingMemoryService emits, so the caller can render them.
        """
        import sys
        import time
        from datetime import datetime

        sys.path.insert(0, "/root")
        from streaming_memory.config import (
            AssistantConfig,
            MemoryConfig,
            ModelConfig,
        )
        from streaming_memory.service import StreamingMemoryService
        from streaming_memory.supermemory_pool import SupermemoryPool

        pool = SupermemoryPool(container_tag=container_tag)

        yield ("status", {"msg": f"ingesting {len(memories)} memories → {container_tag}"})
        t0 = time.time()
        for m in memories:
            created_at = None
            raw = m.get("created_at")
            if raw:
                try:
                    created_at = datetime.fromisoformat(raw.replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    pass
            pool.add(
                content=m["content"],
                emotional_intensity=float(m.get("emotional_intensity", 0.5)),
                created_at=created_at,
            )
        yield ("status", {"msg": f"ingested in {time.time() - t0:.1f}s, letting index settle"})
        time.sleep(3)

        pool_total_tokens = sum(
            len(self.tokenizer.encode(m.content)) for m in pool.memories.values()
        )

        config = AssistantConfig(
            name="supermemory-test",
            system_prompt=system_prompt,
            memory=MemoryConfig(
                memory_file="",
                memory_prefix="[Memories from Supermemory:]",
            ),
            model=ModelConfig(
                model_id=MODEL_ID,
                temperature=0.7,
                max_tokens=max_output_tokens,
            ),
        )

        service = StreamingMemoryService(
            config=config,
            pool=pool,
            tokenizer=self.tokenizer,
            model=self.model,
            pool_total_tokens=pool_total_tokens,
        )

        yield ("status", {"msg": "streaming"})
        try:
            for event in service.generate_stream(
                message=question,
                history=[],
                update_every_n=update_every_n,
                max_memories=max_memories,
                lookback_tokens=lookback_tokens,
            ):
                yield (event.type, event.data)
        finally:
            if cleanup:
                try:
                    deleted = pool.clear_remote()
                    yield ("status", {"msg": f"cleaned up {deleted} docs from {container_tag}"})
                except Exception as e:
                    yield ("status", {"msg": f"cleanup failed: {e}"})

        yield (
            "supermemory_stats",
            {
                "total_searches": pool.total_searches,
                "last_search_ms": pool.last_search_ms,
                "container_tag": container_tag,
            },
        )


# --------------------------------------------------------------------- render


BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
MAGENTA = "\x1b[35m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
RESET = "\x1b[0m"


def _render_event(etype: str, data: dict, state: dict) -> None:
    import sys

    if etype == "status":
        print(f"{DIM}[modal] {data.get('msg', '')}{RESET}")
        return

    if etype == "memories":
        mems = data.get("memories", [])
        print(f"\n{BOLD}== initial memories ({len(mems)}) =={RESET}")
        for i, c in enumerate(mems):
            snippet = c.replace("\n", " ")
            if len(snippet) > 140:
                snippet = snippet[:137] + "..."
            print(f"  {CYAN}• {snippet}{RESET}")
        print(f"\n{BOLD}== streaming (thinking → response) =={RESET}")
        return

    if etype == "thinking":
        if not state["in_think"]:
            sys.stdout.write(f"\n{DIM}<think>{RESET}\n")
            state["in_think"] = True
            state["in_response"] = False
        sys.stdout.write(f"{DIM}{data.get('t', '')}{RESET}")
        sys.stdout.flush()
        return

    if etype == "token":
        if not state["in_response"]:
            if state["in_think"]:
                sys.stdout.write(f"\n{DIM}</think>{RESET}\n\n")
            state["in_response"] = True
            state["in_think"] = False
        sys.stdout.write(data.get("t", ""))
        sys.stdout.flush()
        return

    if etype == "memory_update":
        added = data.get("added", [])
        removed = data.get("removed", [])
        mems = data.get("memories", [])
        retrieve_ms = data.get("retrieve_ms", 0)
        state["swaps"] += 1
        print(
            f"\n\n{BOLD}-- memory swap #{state['swaps']} "
            f"(supermemory {retrieve_ms}ms) --{RESET}"
        )
        for c in added:
            snippet = c.replace("\n", " ")
            if len(snippet) > 140:
                snippet = snippet[:137] + "..."
            print(f"  {GREEN}+ {snippet}{RESET}")
        for c in removed:
            snippet = c.replace("\n", " ")
            if len(snippet) > 140:
                snippet = snippet[:137] + "..."
            print(f"  {RED}- {snippet}{RESET}")
        print(f"  {DIM}now holding {len(mems)} memories{RESET}")
        state["in_think"] = False
        state["in_response"] = False
        return

    if etype == "timing":
        stage = data.get("stage")
        ms = data.get("ms", 0)
        if stage == "embed":
            print(f"{DIM}[timing] initial supermemory search: {ms}ms{RESET}")
        return

    if etype == "max_tokens":
        print(f"\n{YELLOW}[hit max_tokens={data.get('limit')}]{RESET}")
        return

    if etype == "supermemory_stats":
        print(f"\n{BOLD}== supermemory stats =={RESET}")
        print(
            f"  total searches: {data['total_searches']}, "
            f"last: {data['last_search_ms']}ms, "
            f"container_tag: {data['container_tag']}"
        )
        return

    if etype == "done":
        print(f"\n\n{BOLD}== done — {state['swaps']} memory swaps =={RESET}\n")
        return

    if etype in {"context_size", "context_update", "timeline", "thinking_prefix"}:
        return

    print(f"{DIM}[{etype}] {data}{RESET}")


@app.local_entrypoint()
def main(
    question: str = "Dad's birthday is coming up. What should I get him? Think about what's going on in his life lately.",
    memories_path: str = str(dad_memories_path),
    limit_ingest: int = 30,
    update_every_n: int = 10,
    max_memories: int = 5,
    lookback_tokens: int = 60,
    max_output_tokens: int = 1500,
    container_tag: str | None = None,
    cleanup: bool = True,
    save: str | None = None,
):
    """Local driver. Pass through the knobs to the remote class."""
    with open(memories_path) as f:
        all_memories = json.load(f)

    memories = all_memories[:limit_ingest]
    tag = container_tag or f"streaming-memory-test-{uuid.uuid4().hex[:10]}"

    print(f"{BOLD}container_tag:{RESET} {tag}")
    print(f"{BOLD}ingesting {len(memories)} memories{RESET}")
    print(f"{BOLD}question:{RESET} {question}\n")

    system_prompt = """You are a helpful personal assistant who has access to the user's memories and notes.

You help them think through decisions by drawing on what you know about their life, relationships, and past experiences.

When memories are provided, use them naturally to inform your responses. Make connections between different memories when relevant.

Think step by step in <think>...</think> tags before responding.

Be warm and helpful, like a thoughtful friend who knows them well.

Important: Do not use emojis in your responses."""

    tester = SupermemoryStreamingTester()

    state = {"in_think": False, "in_response": False, "swaps": 0}
    events: list[dict] = []
    for etype, data in tester.run_streaming.remote_gen(
        question=question,
        memories=memories,
        container_tag=tag,
        system_prompt=system_prompt,
        update_every_n=update_every_n,
        max_memories=max_memories,
        lookback_tokens=lookback_tokens,
        max_output_tokens=max_output_tokens,
        cleanup=cleanup,
    ):
        _render_event(etype, data, state)
        if etype in {"memories", "memory_update", "supermemory_stats", "timing"}:
            events.append({"type": etype, "data": data})

    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "question": question,
            "container_tag": tag,
            "ingested": len(memories),
            "update_every_n": update_every_n,
            "max_memories": max_memories,
            "lookback_tokens": lookback_tokens,
            "swaps": state["swaps"],
            "events": events,
        }, indent=2))
        print(f"{DIM}saved summary to {out}{RESET}")
