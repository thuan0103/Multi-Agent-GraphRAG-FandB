import asyncio
import functools
import logging
from typing import Callable, Type

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
):

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e
                    if attempt == max_attempts - 1:
                        break

                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"[{func.__name__}] Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retry in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

            logger.error(f"[{func.__name__}] All {max_attempts} attempts failed")
            raise last_exception

        return wrapper
    return decorator