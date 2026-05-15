import asyncio
import logging
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

SUPPORTED = {".csv", ".pdf", ".docx", ".txt"}


class IngestHandler(FileSystemEventHandler):
    def __init__(self, ingest_fn):
        self.ingest_fn = ingest_fn
        self.queue: asyncio.Queue = asyncio.Queue()

    def on_created(self, event):
        if not event.is_directory:
            path = event.src_path
            if Path(path).suffix.lower() in SUPPORTED:
                logger.info(f"[WatchMode] Detected new file: {path}")
                self.queue.put_nowait(path)

    def on_moved(self, event):
        if not event.is_directory:
            path = event.dest_path
            if Path(path).suffix.lower() in SUPPORTED:
                self.queue.put_nowait(path)


async def start_watcher(watch_dir: str, ingest_fn):
    handler = IngestHandler(ingest_fn)
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()
    logger.info(f"[WatchMode] Watching: {watch_dir}")

    try:
        while True:
            path = await handler.queue.get()
            try:
                await ingest_fn(path)
            except Exception as e:
                logger.error(f"[WatchMode] Error ingesting {path}: {e}")
    finally:
        observer.stop()
        observer.join()
