import asyncio
import logging
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

SUPPORTED = {".csv", ".pdf", ".docx", ".txt"}


class IngestHandler(FileSystemEventHandler):
    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
        self.queue = queue
        self.loop = loop

    def _enqueue(self, path: str):
        if Path(path).suffix.lower() in SUPPORTED:
            logger.info(f"[WatchMode] Detected: {path}")
            # thread-safe: watchdog chạy trong background thread riêng
            self.loop.call_soon_threadsafe(self.queue.put_nowait, path)

    def on_created(self, event):
        if not event.is_directory:
            self._enqueue(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._enqueue(event.dest_path)


async def start_watcher(watch_dir: str, ingest_fn):
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    handler = IngestHandler(queue, loop)

    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()
    logger.info(f"[WatchMode] Watching: {watch_dir}")

    try:
        while True:
            path = await queue.get()
            try:
                await ingest_fn(path)
            except Exception as e:
                logger.error(f"[WatchMode] Error ingesting {path}: {e}")
    finally:
        observer.stop()
        observer.join()
