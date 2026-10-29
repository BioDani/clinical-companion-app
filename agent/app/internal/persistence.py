"""SQLite checkpointer and memory store for the LangGraph API.

Files live on the companion-state volume. This is not MemorySaver and not
the Langfuse Postgres.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

CHECKPOINT_PATH = "/data/checkpoints.sqlite"
MEMORY_PATH = "/data/memories.sqlite"


@asynccontextmanager
async def generate_checkpointer():
    """Async saver the LangGraph server keeps open for the process."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_PATH) as saver:
        await saver.setup()
        yield saver


_memory_store = None


def open_memory_store():
    """Process-wide store of personal notes. The connection stays on its loop."""
    global _memory_store
    if _memory_store is not None:
        return _memory_store
    import asyncio
    import threading

    import aiosqlite
    from langgraph.store.sqlite.aio import AsyncSqliteStore

    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="memory-store", daemon=True).start()

    async def _open() -> AsyncSqliteStore:
        conn = await aiosqlite.connect(MEMORY_PATH, isolation_level=None)
        store = AsyncSqliteStore(conn)
        await store.setup()
        return store

    _memory_store = asyncio.run_coroutine_threadsafe(_open(), loop).result()
    return _memory_store
